from fastapi import FastAPI, HTTPException
from .ml import run_forecast
from .settings import RUN_TOKEN

app = FastAPI(title="Clima Forecast API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast/run")
def forecast_run(token: str):
    if RUN_TOKEN and token != RUN_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    return run_forecast()

