from fastapi import FastAPI, HTTPException
import pandas as pd
import mysql.connector

from .settings import RUN_TOKEN, DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME
from .ml import run_forecast

app = FastAPI(title="Clima Forecast API")

def db_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS, database=DB_NAME
    )

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/forecast/run")
def forecast_run(token: str):
    if RUN_TOKEN and token != RUN_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    return run_forecast()

@app.get("/forecast/next-year")
def forecast_next_year():
    conn = db_conn()
    q = """
    SELECT *
    FROM clima_forecast_diario
    WHERE fecha BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 365 DAY)
    ORDER BY fecha;
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df.to_dict(orient="records")
