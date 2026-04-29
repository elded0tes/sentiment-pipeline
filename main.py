"""
main.py
-------
Punto de entrada del pipeline. Ejecuta:
    python main.py
    python main.py --company microsoft --pages 5
    python main.py --data data/raw/google_reviews_raw.csv  # omite scraping
"""

import argparse
import logging
from src.pipeline import SentimentPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Teamblind Sentiment Analysis Pipeline")
    parser.add_argument("--company",  type=str,  default="google",    help="Empresa a analizar")
    parser.add_argument("--pages",    type=int,  default=3,           help="Número de páginas a scrapear")
    parser.add_argument("--data",     type=str,  default=None,        help="CSV de datos existentes (omite scraping)")
    parser.add_argument("--tracking", type=str,  default="mlruns",    help="URI de MLFlow tracking")
    parser.add_argument("--no-headless", action="store_true",         help="Mostrar ventana del browser")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline = SentimentPipeline(
        company=args.company,
        max_pages=args.pages,
        data_path=args.data,
        tracking_uri=args.tracking,
        headless=not args.no_headless,
    )
    pipeline.run()
