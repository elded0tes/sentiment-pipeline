"""
pipeline.py
-----------
Orquesta el pipeline completo e integra MLFlow para tracking.
"""

import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from pathlib import Path

from src.scraper import TeamblindScraper
from src.preprocessor import TextPreprocessor
from src.model import SentimentAnalyzer, SentimentClassifier, Visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    Pipeline completo:
      scraping → preprocessing → sentiment (VADER) → classification → MLFlow tracking

    Uso:
        pipeline = SentimentPipeline(company="google", max_pages=3)
        pipeline.run()
    """

    EXPERIMENT_NAME = "Teamblind-Sentiment-Analysis"

    def __init__(
        self,
        company: str = "google",
        max_pages: int = 3,
        data_path: str = None,          # Si ya tienes datos cargados omite el scraping
        tracking_uri: str = "mlruns",   # Carpeta local o URL de un servidor MLFlow
        headless: bool = True,
    ):
        self.company      = company
        self.max_pages    = max_pages
        self.data_path    = data_path
        self.tracking_uri = tracking_uri
        self.headless     = headless

        # Sub-módulos
        self.scraper    = TeamblindScraper(company=company, max_pages=max_pages, headless=headless)
        self.preprocessor = TextPreprocessor()
        self.analyzer   = SentimentAnalyzer()
        self.classifier = SentimentClassifier()
        self.visualizer = Visualizer(output_dir="plots")

        # Configurar MLFlow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.EXPERIMENT_NAME)
        logger.info(f"MLFlow Tracking URI: {mlflow.get_tracking_uri()}")

    # ------------------------------------------------------------------
    # Pasos individuales del pipeline
    # ------------------------------------------------------------------

    def step_scrape(self) -> pd.DataFrame:
        if self.data_path and Path(self.data_path).exists():
            logger.info(f"Cargando datos existentes desde: {self.data_path}")
            return pd.read_csv(self.data_path)
        logger.info("Iniciando scraping de Teamblind...")
        df = self.scraper.run()
        self.scraper.save(df)
        return df

    def step_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocesando texto...")
        df_proc = self.preprocessor.process_dataframe(df, text_col="review")
        self.preprocessor.save(df_proc)
        return df_proc

    def step_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analizando sentimientos con VADER...")
        return self.analyzer.analyze_dataframe(df, text_col="cleaned_text")

    def step_classify(self, df: pd.DataFrame) -> dict:
        logger.info("Entrenando clasificador...")
        return self.classifier.train(
            df,
            text_col="processed_text",
            label_col="sentiment_label",
        )

    def step_visualize(self, df: pd.DataFrame) -> dict[str, str]:
        logger.info("Generando visualizaciones...")
        return self.visualizer.all_plots(df, self.classifier)

    # ------------------------------------------------------------------
    # MLFlow tracking
    # ------------------------------------------------------------------

    def _log_to_mlflow(
        self,
        df: pd.DataFrame,
        metrics: dict,
        plots: dict[str, str],
        params: dict,
    ):
        """Registra parámetros, métricas, artefactos y modelo en MLFlow."""

        with mlflow.start_run(run_name=f"{self.company}-sentiment") as run:
            run_id = run.info.run_id
            logger.info(f"MLFlow Run ID: {run_id}")

            # --- Parámetros del experimento ---
            mlflow.log_params(params)

            # --- Métricas del modelo ---
            mlflow.log_metrics({
                "accuracy":        metrics.get("accuracy", 0),
                "f1_weighted":     metrics.get("f1_weighted", 0),
                "f1_macro":        metrics.get("f1_macro", 0),
                "cv_f1_mean":      metrics.get("cv_f1_mean", 0),
                "cv_f1_std":       metrics.get("cv_f1_std", 0),
            })

            # --- Métricas de datos ---
            sentiment_counts = df["sentiment_label"].value_counts().to_dict()
            mlflow.log_metrics({
                "n_reviews":         len(df),
                "n_positive":        sentiment_counts.get("Positive", 0),
                "n_neutral":         sentiment_counts.get("Neutral", 0),
                "n_negative":        sentiment_counts.get("Negative", 0),
                "avg_compound_score": float(df["compound"].mean()),
                "avg_token_count":    float(df["token_count"].mean()),
            })

            # --- Artefactos: plots ---
            for name, path in plots.items():
                mlflow.log_artifact(path, artifact_path="plots")
                logger.info(f"  ↑ Artifact: {name}")

            # --- Artefacto: datos procesados ---
            mlflow.log_artifact("data/processed/reviews_processed.csv", artifact_path="data")

            # --- Modelo con firma ---
            sample_input  = df["processed_text"].head(5).tolist()
            sample_output = self.classifier.predict(sample_input)
            signature = infer_signature(
                {"text": sample_input},
                {"prediction": sample_output.tolist()},
            )
            mlflow.sklearn.log_model(
                sk_model=self.classifier.pipeline,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"TeamblindSentiment-{self.company.capitalize()}",
            )
            logger.info("Modelo registrado en MLFlow.")

            # Guardar resultados finales con sentimiento
            results_path = "data/processed/reviews_with_sentiment.csv"
            df.drop(columns=["tokens"], errors="ignore").to_csv(results_path, index=False)
            mlflow.log_artifact(results_path, artifact_path="data")

            tracking_url = f"{mlflow.get_tracking_uri()}/#/experiments"
            logger.info(f"\n{'='*60}")
            logger.info(f"✅  MLFlow Tracking URL: {tracking_url}")
            logger.info(f"   Run ID: {run_id}")
            logger.info(f"{'='*60}")
            return run_id

    # ------------------------------------------------------------------
    # Orquestador principal
    # ------------------------------------------------------------------

    def run(self):
        """Ejecuta el pipeline completo."""
        logger.info("\n" + "="*60)
        logger.info("  INICIANDO PIPELINE DE ANÁLISIS SENTIMENTAL")
        logger.info("="*60)

        # 1. Scraping
        df_raw = self.step_scrape()
        if df_raw.empty:
            logger.error("No se obtuvieron datos. Abortando pipeline.")
            return

        # 2. Preprocesamiento
        df_proc = self.step_preprocess(df_raw)

        # 3. Análisis de sentimientos
        df_sentiment = self.step_sentiment(df_proc)

        # 4. Clasificación
        metrics = self.step_classify(df_sentiment)

        # 5. Visualizaciones
        plots = self.step_visualize(df_sentiment)

        # 6. MLFlow tracking
        params = {
            "company":       self.company,
            "max_pages":     self.max_pages,
            "max_features":  self.classifier.max_features,
            "ngram_range":   str(self.classifier.ngram_range),
            "test_size":     self.classifier.test_size,
            "model_type":    "TF-IDF + LogisticRegression",
            "sentiment_method": "VADER",
        }
        run_id = self._log_to_mlflow(df_sentiment, metrics, plots, params)

        logger.info("\n✅ Pipeline completado exitosamente.")
        logger.info(f"   Ejecuta: mlflow ui --port 5000")
        logger.info(f"   Abre:    http://localhost:5000\n")
        return df_sentiment, metrics, run_id
