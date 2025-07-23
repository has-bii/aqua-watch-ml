
import os
from dotenv import load_dotenv
from pydantic import Field
from pathlib import Path

load_dotenv()

class Settings:
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Supabase configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL") # type: ignore
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY") # type: ignore

    # RabbitMQ configuration
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL") # type: ignore
    RABBITMQ_HOST: str = os.getenv("RABBITMQ_HOST", "localhost") # type: ignore
    RABBITMQ_PORT: int = int(os.getenv("RABBITMQ_PORT", 5672)) # type: ignore
    RABBITMQ_VHOST: str = os.getenv("RABBITMQ_VHOST", "/") # type: ignore
    RABBITMQ_USER: str = os.getenv("RABBITMQ_USER", "guest") # type: ignore
    RABBITMQ_PASSWORD: str = os.getenv("RABBITMQ_PASSWORD", "guest") # type: ignore

    # ML Settings
    ML_MODEL_RETRAIN_INTERVAL = 24  # hours
    ANOMALY_DETECTION_INTERVAL = 30  # minutes
    PREDICTION_INTERVAL = 30  # minutes
    
    # Data retention and processing
    MEASUREMENTS_PER_HOUR = 12  # 5-minute intervals = 12 per hour
    MEASUREMENTS_PER_DAY = 288  # 12 * 24
    FEATURE_LOOKBACK_HOURS = 2  # Reduce lookback for faster processing
    DATA_INTERVAL_MINUTES = 5  # Data collection interval in minutes
    
    # Directories
    MODEL_SAVE_PATH = "models/saved_models"
    LOG_PATH = "logs"
    DATA_CACHE_PATH = "data/cache"

    model_parameters: dict = Field(default_factory=lambda: {
        "n_estimators": 150,
        "max_depth": 20,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1
    })
    model_save_path = Path(MODEL_SAVE_PATH)

settings = Settings()