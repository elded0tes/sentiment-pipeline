"""
preprocessor.py
---------------
Limpieza, normalización y preparación del texto para el modelado.
"""

import re
import logging
import pandas as pd
import numpy as np
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_nltk_resources():
    """Descarga recursos NLTK necesarios (solo primera vez)."""
    resources = ["stopwords", "wordnet", "punkt", "averaged_perceptron_tagger", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logger.warning(f"No se pudo descargar {resource}: {e}")


download_nltk_resources()


class TextPreprocessor:
    """
    Pipeline de preprocesamiento de texto para análisis de sentimientos.

    Pasos:
        1. Limpieza básica (HTML, URLs, caracteres especiales)
        2. Normalización (lowercase, espacios)
        3. Tokenización
        4. Eliminación de stop words
        5. Lematización
        6. Creación del DataFrame final
    """

    def __init__(
        self,
        language: str = "english",
        min_text_length: int = 10,
        extra_stopwords: list[str] = None,
    ):
        self.language = language
        self.min_text_length = min_text_length
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))

        # Stop words adicionales del dominio
        domain_stopwords = {"company", "work", "job", "team", "people", "good", "great"}
        if extra_stopwords:
            domain_stopwords.update(extra_stopwords)
        self.stop_words.update(domain_stopwords)

        logger.info(f"TextPreprocessor inicializado (idioma: {language})")

    # ------------------------------------------------------------------
    # Limpieza de texto
    # ------------------------------------------------------------------

    def _remove_html(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", " ", text)

    def _remove_special_chars(self, text: str) -> str:
        # Conservar signos de puntuación básicos que ayudan al sentimiento
        return re.sub(r"[^a-zA-Z0-9\s!?.,']", " ", text)

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def clean_text(self, text: str) -> str:
        """Aplica todos los pasos de limpieza."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = self._remove_special_chars(text)
        text = self._normalize_whitespace(text)
        return text

    # ------------------------------------------------------------------
    # Tokenización y normalización léxica
    # ------------------------------------------------------------------

    def tokenize_and_filter(self, text: str) -> list[str]:
        """Tokeniza, elimina stop words y lematiza."""
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
            and len(token) > 2
            and token.isalpha()
        ]
        return tokens

    def process_text(self, text: str) -> dict:
        """Procesa un texto y retorna texto limpio + tokens."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return {
            "cleaned_text": cleaned,
            "tokens": tokens,
            "token_count": len(tokens),
            "processed_text": " ".join(tokens),
        }

    # ------------------------------------------------------------------
    # Procesamiento de DataFrame
    # ------------------------------------------------------------------

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
        """
        Toma el DataFrame crudo y retorna uno enriquecido con
        columnas de texto preprocesado.
        """
        logger.info(f"Preprocesando {len(df)} registros...")
        df = df.copy()

        # Combinar título + reseña si está disponible
        if "title" in df.columns:
            df["full_text"] = df["title"].fillna("") + ". " + df[text_col].fillna("")
        else:
            df["full_text"] = df[text_col].fillna("")

        # Aplicar procesamiento
        processed = df["full_text"].apply(self.process_text)
        df["cleaned_text"]   = processed.apply(lambda x: x["cleaned_text"])
        df["tokens"]         = processed.apply(lambda x: x["tokens"])
        df["token_count"]    = processed.apply(lambda x: x["token_count"])
        df["processed_text"] = processed.apply(lambda x: x["processed_text"])

        # Normalizar rating a numérico (1–5)
        if "rating" in df.columns:
            df["rating_num"] = pd.to_numeric(
                df["rating"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
                errors="coerce",
            )

        # Filtrar textos demasiado cortos
        before = len(df)
        df = df[df["token_count"] >= self.min_text_length // 3].reset_index(drop=True)
        logger.info(f"Filas eliminadas por texto muy corto: {before - len(df)}")
        logger.info(f"Dataset final: {len(df)} registros.")
        return df

    # ------------------------------------------------------------------
    # Guardar
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_dir: str = "data/processed") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = f"{output_dir}/reviews_processed.csv"
        df.drop(columns=["tokens"], errors="ignore").to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Datos procesados guardados en: {path}")
        return path
