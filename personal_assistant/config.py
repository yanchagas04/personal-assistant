import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Caminhos do projeto
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Configurações do Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
DEFAULT_BASE_MODEL = os.getenv("GEMINI_BASE_MODEL", "gemini-1.5-flash-001")


def get_gemini_api_key() -> str:
    """Retorna a chave da API do Gemini ou levanta um erro explicativo caso não esteja definida."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "Chave de API do Gemini não encontrada! "
            "Por favor, adicione a variável GEMINI_API_KEY no seu arquivo .env na raiz do projeto.\n"
            "Exemplo no .env:\n"
            "GEMINI_API_KEY=AIzaSy..."
        )
    return GEMINI_API_KEY


# Se tqdm estiver instalado, configura loguru
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
