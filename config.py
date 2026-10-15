import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از فایل .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

# تنظیمات لاگینگ استاندارد
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TennisV1")


class Config:
    """
    تنظیمات مرکزی ربات هوشمند کوانت و روانشناسی تنیس — ورژن ۱
    """
    VERSION_NAME: str = "ورژن 1"
    APP_TITLE: str = "ربات هوشمند تحلیل و پیش‌بینی کوانت تنیس (ورژن ۱)"

    # --- تنظیمات تلگرام ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    ENABLE_TELEGRAM_LISTENER: bool = os.getenv("ENABLE_TELEGRAM_LISTENER", "true").lower() in ("true", "1", "yes")

    # --- تنظیمات مدل هوش مصنوعی (NVIDIA NIM) ---
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "moonshotai/kimi-k3").strip()
    LLM_FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "deepseek-ai/deepseek-v4.1-flash").strip()

    # --- پارامترهای شبیه‌سازی و وسعت اسکن روزانه ---
    MONTE_CARLO_SIMULATIONS: int = int(os.getenv("MONTE_CARLO_SIMULATIONS", "10000"))
    MAX_DAILY_MATCHES: int = int(os.getenv("MAX_DAILY_MATCHES", "25"))
    RECENT_MATCHES_TELEMETRY: int = int(os.getenv("RECENT_MATCHES_TELEMETRY", "7"))

    # --- گیت‌های کیفیت، دقت و کنترل ریسک (Quality Over Quantity) ---
    MIN_PROB_SPREAD: float = float(os.getenv("MIN_PROB_SPREAD", "9.0"))
    MAX_SHANNON_ENTROPY: float = float(os.getenv("MAX_SHANNON_ENTROPY", "0.988"))
    MIN_LLM_CONFIDENCE: float = float(os.getenv("MIN_LLM_CONFIDENCE", "60.0"))
    STEAM_MOVE_ALERT_PCT: float = float(os.getenv("STEAM_MOVE_ALERT_PCT", "5.5"))

    # --- مسیر دیتابیس و کش ---
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./tennis_data_cache"))
    DB_PATH: Path = Path(os.getenv("DB_PATH", "./tennis_data_cache/tennis_v1_ledger.db"))

    # --- زمان‌بندی روزانه ---
    MORNING_RUN_TIME: str = os.getenv("MORNING_RUN_TIME", "09:30").strip()
    NIGHT_SETTLE_TIME: str = os.getenv("NIGHT_SETTLE_TIME", "23:30").strip()
