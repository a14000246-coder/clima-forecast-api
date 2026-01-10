import os

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

RUN_TOKEN = os.getenv("RUN_TOKEN", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "prophet-v1")
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "365"))
