"""
Central configuration for CreditIQ.
All settings are loaded from environment variables (via .env file).
Import this anywhere: from config import settings
"""

from pathlib import Path

from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    app_name: str = "CreditIQ"
    app_secret_key: str = "change-me-in-production"

    # Database (SQL Server)
    database_url: str = (
        "mssql+pyodbc://sa:CreditIQ_Pass123!@localhost:1433/creditiq_db"
        "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )

    # Redis
    redis_url: str = "redis://localhost:6379"

    # JWT
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # External APIs
    fred_api_key: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""

    # MLflow
    mlflow_tracking_uri: str = "./mlflow_runs"
    mlflow_experiment_name: str = "creditiq"

    # Model paths
    xgboost_model_path: str = "./models/xgboost/model.pkl"
    finbert_model_path: str = "./models/finbert/"
    lstm_model_path: str = "./models/lstm/model.pt"
    ensemble_model_path: str = "./models/ensemble/model.pkl"

    # Data paths
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    external_data_dir: Path = ROOT_DIR / "data" / "external"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
