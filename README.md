# 🔍 Teamblind Sentiment Analysis Pipeline

Pipeline completo de NLP/MLOps para analizar el sentimiento de reseñas de empresas en **Teamblind**, desde el scraping hasta el tracking con MLFlow.

---

## 🗂️ Estructura del Proyecto

```
sentiment-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD con GitHub Actions
├── data/
│   ├── raw/                    # Datos scrapeados (gitignored)
│   └── processed/              # Datos preprocesados (gitignored)
├── plots/                      # Visualizaciones generadas
├── notebooks/
│   └── exploration.ipynb       # EDA exploratorio
├── src/
│   ├── __init__.py
│   ├── scraper.py              # Selenium scraper para Teamblind
│   ├── preprocessor.py         # Limpieza y normalización de texto
│   ├── model.py                # VADER + TF-IDF + LogReg + plots
│   └── pipeline.py             # Orquestador + MLFlow tracking
├── .gitignore
├── main.py                     # Punto de entrada
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

### 1. Clonar y crear entorno virtual
```bash
git clone https://github.com/TU_USUARIO/sentiment-pipeline.git
cd sentiment-pipeline

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Instalar Google Chrome
El scraper necesita Chrome (la versión correcta del driver se instala automáticamente).
- macOS: `brew install --cask google-chrome`
- Linux: `sudo apt-get install -y google-chrome-stable`

---

## 🚀 Ejecución

### Ejecutar pipeline completo (con scraping)
```bash
python main.py --company google --pages 5
```

### Ejecutar con datos existentes (omite scraping)
```bash
python main.py --data data/raw/reviews_raw.csv
```

### Ver resultados en MLFlow UI
```bash
mlflow ui --port 5000
# Abre: http://localhost:5000
```

### Argumentos disponibles
| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--company` | Empresa a analizar | `google` |
| `--pages` | Páginas a scrapear | `3` |
| `--data` | CSV de datos locales | `None` |
| `--tracking` | URI de MLFlow | `mlruns` |
| `--no-headless` | Mostrar ventana del browser | `False` |

---

## 📦 Módulos

### `src/scraper.py` — Web Scraping
- Usa **Selenium + BeautifulSoup** para extraer reseñas de Teamblind
- Maneja paginación, rate limiting y renderizado JS
- Guarda datos en `data/raw/{company}_reviews_raw.csv`

### `src/preprocessor.py` — Preprocesamiento
- Limpieza: HTML, URLs, caracteres especiales
- Normalización: lowercase, espacios
- Tokenización, stop words, lematización (NLTK)
- Crea columnas: `cleaned_text`, `processed_text`, `token_count`

### `src/model.py` — Modelado
- **SentimentAnalyzer**: VADER (sin entrenamiento, basado en léxico)
  - Produce: `compound`, `positive_score`, `negative_score`, `sentiment_label`
- **SentimentClassifier**: TF-IDF + Logistic Regression
  - Cross-validation 5-fold, métricas de accuracy y F1
- **Visualizer**: genera 5 plots y los guarda en `plots/`

### `src/pipeline.py` — Orquestador MLFlow
- Conecta todos los pasos
- Registra con MLFlow: parámetros, métricas, artefactos, firma del modelo

---

## 📊 MLFlow — Qué se registra

| Categoría | Detalle |
|-----------|---------|
| **Parámetros** | company, pages, max_features, ngram_range, model_type |
| **Métricas del modelo** | accuracy, f1_weighted, f1_macro, cv_f1_mean, cv_f1_std |
| **Métricas de datos** | n_reviews, n_positive, n_negative, avg_compound |
| **Artefactos** | 5 plots PNG, CSV procesado, CSV con sentimiento |
| **Modelo** | Pipeline sklearn serializado con firma (MLFlow Model Registry) |

---

## 🔁 GitHub Actions (CI/CD)

El workflow `.github/workflows/ci.yml` se activa en cada push a `main` o PR:
1. **Lint** con flake8
2. **Tests** con pytest
3. **MLFlow run** automático (solo en `main`)
4. **Upload** de artefactos MLFlow como GitHub Artifacts

---

## 📝 Notas importantes

- **Teamblind** requiere JavaScript; si el scraper no extrae reseñas, prueba `--no-headless` para inspeccionar visualmente el browser.
- Los selectores CSS del scraper pueden cambiar; revisa `_parse_reviews()` si cambia el diseño del sitio.
- Para producción, considera usar un **servidor MLFlow remoto** (ej. en AWS/GCP).

---

## 🛠️ Stack Tecnológico

| Herramienta | Uso |
|-------------|-----|
| Selenium + BS4 | Web scraping |
| NLTK | Tokenización, stop words, lematización |
| VADER | Análisis de sentimientos léxico |
| scikit-learn | TF-IDF + Logistic Regression |
| MLFlow | Tracking de experimentos y model registry |
| GitHub Actions | CI/CD automatizado |
