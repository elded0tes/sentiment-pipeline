"""
scraper.py
----------
Extrae reseñas de Teamblind usando Selenium + BeautifulSoup.
Teamblind usa renderizado JS, por lo que se requiere Selenium.
"""

import time
import logging
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TeamblindScraper:
    """
    Scraper para obtener reseñas de empresas en Teamblind.

    Uso:
        scraper = TeamblindScraper(company="google", max_pages=5)
        df = scraper.run()
    """

    BASE_URL = "https://www.teamblind.com/company/{company}/reviews?page={page}"

    def __init__(self, company: str = "google", max_pages: int = 5, headless: bool = True):
        self.company = company.lower().replace(" ", "-")
        self.max_pages = max_pages
        self.headless = headless
        self.driver = None

    # ------------------------------------------------------------------
    # Driver setup
    # ------------------------------------------------------------------

    def _init_driver(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        logger.info("Driver inicializado correctamente.")

    def _close_driver(self):
        if self.driver:
            self.driver.quit()
            logger.info("Driver cerrado.")

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def _get_page_source(self, url: str) -> str:
        self.driver.get(url)
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.review-card, article"))
            )
        except Exception:
            logger.warning(f"Timeout esperando contenido en: {url}")
        time.sleep(2)  # buffer extra para JS dinámico
        return self.driver.page_source

    def _parse_reviews(self, html: str) -> list[dict]:
        """Parsea las reseñas de una página HTML."""
        soup = BeautifulSoup(html, "html.parser")
        reviews = []

        # Teamblind puede cambiar sus selectores; ajusta si es necesario
        cards = soup.select("div[class*='review'], article[class*='review']")

        for card in cards:
            try:
                title_el   = card.select_one("h3, [class*='title'], [class*='heading']")
                body_el    = card.select_one("p, [class*='body'], [class*='content']")
                rating_el  = card.select_one("[class*='rating'], [class*='star']")
                date_el    = card.select_one("time, [class*='date']")
                role_el    = card.select_one("[class*='role'], [class*='job'], [class*='position']")

                review = {
                    "title":  title_el.get_text(strip=True)  if title_el  else "",
                    "review": body_el.get_text(strip=True)   if body_el   else "",
                    "rating": rating_el.get_text(strip=True) if rating_el else "",
                    "date":   date_el.get("datetime", date_el.get_text(strip=True)) if date_el else "",
                    "role":   role_el.get_text(strip=True)   if role_el   else "",
                    "company": self.company,
                }

                # Solo guardar si tiene contenido real
                if review["review"] or review["title"]:
                    reviews.append(review)

            except Exception as e:
                logger.debug(f"Error parseando card: {e}")
                continue

        return reviews

    def run(self) -> pd.DataFrame:
        """Ejecuta el scraper y retorna un DataFrame."""
        self._init_driver()
        all_reviews = []

        try:
            for page in range(1, self.max_pages + 1):
                url = self.BASE_URL.format(company=self.company, page=page)
                logger.info(f"Scrapeando página {page}: {url}")

                html = self._get_page_source(url)
                reviews = self._parse_reviews(html)

                if not reviews:
                    logger.info(f"Sin reseñas en página {page}. Terminando.")
                    break

                all_reviews.extend(reviews)
                logger.info(f"  → {len(reviews)} reseñas encontradas (total: {len(all_reviews)})")
                time.sleep(1.5)  # respetar el rate limiting

        finally:
            self._close_driver()

        df = pd.DataFrame(all_reviews)
        logger.info(f"Scraping completo. Total de reseñas: {len(df)}")
        return df

    # ------------------------------------------------------------------
    # Guardar datos
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, output_dir: str = "data/raw") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = f"{output_dir}/{self.company}_reviews_raw.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Datos guardados en: {path}")
        return path
