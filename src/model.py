"""
model.py
--------
Módulo de modelado:
  1. Análisis de sentimientos con VADER (léxico, sin entrenamiento)
  2. Clasificación supervisada con TF-IDF + Logistic Regression
  3. Generación de visualizaciones para MLFlow
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Análisis de sentimientos (VADER — sin entrenamiento)
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Analiza el sentimiento de reseñas usando VADER.
    Produce: compound score, etiqueta (Positive / Neutral / Negative)
    y probabilidades individuales.
    """

    THRESHOLD_POS =  0.05
    THRESHOLD_NEG = -0.05

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("SentimentAnalyzer (VADER) inicializado.")

    def analyze(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= self.THRESHOLD_POS:
            label = "Positive"
        elif compound <= self.THRESHOLD_NEG:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "compound":      compound,
            "positive_score": scores["pos"],
            "negative_score": scores["neg"],
            "neutral_score":  scores["neu"],
            "sentiment_label": label,
        }

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
        logger.info("Calculando sentimientos con VADER...")
        df = df.copy()
        results = df[text_col].apply(self.analyze)
        sentiment_df = pd.DataFrame(results.tolist())
        df = pd.concat([df, sentiment_df], axis=1)
        logger.info(f"Distribución de sentimientos:\n{df['sentiment_label'].value_counts()}")
        return df


# ---------------------------------------------------------------------------
# Clasificación supervisada (TF-IDF + Logistic Regression)
# ---------------------------------------------------------------------------

class SentimentClassifier:
    """
    Clasificador supervisado basado en TF-IDF + Logistic Regression.

    Si el DataFrame ya tiene 'sentiment_label' (de VADER), lo usa como target.
    En producción puedes sustituir por etiquetas humanas.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.max_features  = max_features
        self.ngram_range   = ngram_range
        self.test_size     = test_size
        self.random_state  = random_state
        self.pipeline      = None
        self.metrics       = {}
        self.classes_      = None
        logger.info("SentimentClassifier inicializado.")

    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=self.random_state,
                C=1.0,
            )),
        ])

    def train(
        self,
        df: pd.DataFrame,
        text_col: str = "processed_text",
        label_col: str = "sentiment_label",
    ) -> dict:
        """Entrena el clasificador y calcula métricas."""

        X = df[text_col].fillna("").values
        y = df[label_col].values
        self.classes_ = sorted(df[label_col].unique())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.pipeline = self._build_pipeline()
        logger.info("Entrenando modelo...")
        self.pipeline.fit(X_train, y_train)

        # Predicciones
        y_pred = self.pipeline.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="f1_weighted")

        # Métricas
        self.metrics = {
            "accuracy":        float(accuracy_score(y_test, y_pred)),
            "f1_weighted":     float(f1_score(y_test, y_pred, average="weighted")),
            "f1_macro":        float(f1_score(y_test, y_pred, average="macro")),
            "cv_f1_mean":      float(cv_scores.mean()),
            "cv_f1_std":       float(cv_scores.std()),
            "n_train":         len(X_train),
            "n_test":          len(X_test),
            "max_features":    self.max_features,
            "ngram_range":     str(self.ngram_range),
        }

        logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"F1 Weighted: {self.metrics['f1_weighted']:.4f}")
        logger.info(f"CV F1: {self.metrics['cv_f1_mean']:.4f} ± {self.metrics['cv_f1_std']:.4f}")

        # Guardar para plots
        self._y_test  = y_test
        self._y_pred  = y_pred
        self._X_test  = X_test
        return self.metrics

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

class Visualizer:
    """Genera y guarda plots para MLFlow."""

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def plot_sentiment_distribution(self, df: pd.DataFrame, label_col: str = "sentiment_label") -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Distribución de Sentimientos", fontsize=15, fontweight="bold")

        counts = df[label_col].value_counts()
        colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        bar_colors = [colors.get(c, "#3498db") for c in counts.index]

        # Bar chart
        axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=1.2)
        axes[0].set_title("Conteo por Sentimiento")
        axes[0].set_ylabel("Número de Reseñas")
        for i, (idx, val) in enumerate(counts.items()):
            axes[0].text(i, val + 0.5, str(val), ha="center", fontsize=11, fontweight="bold")

        # Pie chart
        axes[1].pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=bar_colors, startangle=140, textprops={"fontsize": 11}
        )
        axes[1].set_title("Proporción por Sentimiento")

        plt.tight_layout()
        path = str(self.output_dir / "sentiment_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot guardado: {path}")
        return path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list,
    ) -> str:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax, linewidths=0.5
        )
        ax.set_title("Matriz de Confusión", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicción", fontsize=12)
        ax.set_ylabel("Real", fontsize=12)
        plt.tight_layout()
        path = str(self.output_dir / "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot guardado: {path}")
        return path

    def plot_compound_distribution(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        for label, group in df.groupby("sentiment_label"):
            ax.hist(
                group["compound"], bins=30, alpha=0.6,
                label=label, color=colors.get(label, "blue")
            )
        ax.axvline(0.05, color="green", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(-0.05, color="red",   linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title("Distribución del Compound Score (VADER)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Compound Score")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        plt.tight_layout()
        path = str(self.output_dir / "compound_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_wordcloud(self, df: pd.DataFrame, text_col: str = "processed_text") -> str:
        all_text = " ".join(df[text_col].fillna("").values)
        wc = WordCloud(
            width=900, height=450, background_color="white",
            colormap="RdYlGn", max_words=200,
        ).generate(all_text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud — Reseñas Teamblind", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = str(self.output_dir / "wordcloud.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_top_features(self, classifier: SentimentClassifier, top_n: int = 15) -> str:
        """Top features por clase para el modelo TF-IDF."""
        tfidf = classifier.pipeline.named_steps["tfidf"]
        clf   = classifier.pipeline.named_steps["clf"]
        feature_names = tfidf.get_feature_names_out()
        classes = clf.classes_

        n_classes = len(classes)
        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6))
        if n_classes == 1:
            axes = [axes]

        colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}

        for ax, cls, coef in zip(axes, classes, clf.coef_):
            top_idx = coef.argsort()[-top_n:]
            top_features = feature_names[top_idx]
            top_weights  = coef[top_idx]
            color = colors.get(cls, "#3498db")
            ax.barh(top_features, top_weights, color=color, alpha=0.85)
            ax.set_title(f"Top Features: {cls}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Coeficiente TF-IDF")

        plt.suptitle("Features más importantes por clase", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = str(self.output_dir / "top_features.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def all_plots(
        self,
        df: pd.DataFrame,
        classifier: SentimentClassifier,
    ) -> dict[str, str]:
        """Genera y retorna todos los plots."""
        plots = {
            "sentiment_distribution": self.plot_sentiment_distribution(df),
            "compound_distribution":  self.plot_compound_distribution(df),
            "wordcloud":              self.plot_wordcloud(df),
        }
        if hasattr(classifier, "_y_test"):
            plots["confusion_matrix"] = self.plot_confusion_matrix(
                classifier._y_test,
                classifier._y_pred,
                classifier.classes_,
            )
        if classifier.pipeline is not None:
            plots["top_features"] = self.plot_top_features(classifier)
        return plots
