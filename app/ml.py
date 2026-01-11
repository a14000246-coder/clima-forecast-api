import os
from ftplib import FTP
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from prophet import Prophet

from .settings import MODEL_VERSION, HORIZON_DAYS


def fit_predict(df: pd.DataFrame, col: str, non_negative=False, cap99=False, log1p=False) -> pd.DataFrame:
    """
    Entrena Prophet con una serie diaria y devuelve HORIZON_DAYS predicciones (yhat) para el futuro.
    df debe tener columnas: fecha (datetime) y col (numérico).
    """
    data = df[["fecha", col]].dropna().copy()
    data.rename(columns={"fecha": "ds", col: "y"}, inplace=True)

    if log1p:
        data["y"] = np.log1p(np.maximum(data["y"].astype(float), 0))

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(data)

    future = m.make_future_dataframe(periods=HORIZON_DAYS, freq="D")
    fc = m.predict(future)[["ds", "yhat"]].tail(HORIZON_DAYS).copy()

    yhat = fc["yhat"].to_numpy()
    if log1p:
        yhat = np.expm1(yhat)
    if non_negative:
        yhat = np.maximum(yhat, 0)
    if cap99:
        yhat = np.minimum(yhat, 99.99)

    fc["fecha"] = pd.to_datetime(fc["ds"]).dt.date
    fc["yhat"] = np.round(yhat, 2)
    return fc[["fecha", "yhat"]]


def upload_csv_ftp(local_file: str):
    ftp_host = os.getenv("FTP_HOST")
    ftp_port = int(os.getenv("FTP_PORT", "21"))
    ftp_user = os.getenv("FTP_USER")
    ftp_pass = os.getenv("FTP_PASS")
    ftp_dir = os.getenv("FTP_REMOTE_DIR")

    if not all([ftp_host, ftp_user, ftp_pass, ftp_dir]):
        raise RuntimeError("Faltan variables FTP_* (FTP_HOST/FTP_USER/FTP_PASS/FTP_REMOTE_DIR).")

    ftp = FTP()
    ftp.connect(ftp_host, ftp_port, timeout=30)
    ftp.login(ftp_user, ftp_pass)

    # Si falla aquí, la ruta remota no existe o no coincide con tu hosting
    ftp.cwd(ftp_dir)

    with open(local_file, "rb") as f:
        ftp.storbinary("STOR forecast_1y.csv", f)

    ftp.quit()


def trigger_php_import():
    url = os.getenv("IMPORT_URL")
    token = os.getenv("IMPORT_TOKEN")

    if not url or not token:
        raise RuntimeError("Faltan variables IMPORT_URL / IMPORT_TOKEN.")

    r = requests.get(url, params={"token": token}, timeout=120)
    r.raise_for_status()
    return r.text


def run_forecast():
    """
    Lee el histórico diario desde un CSV local (data/clima_diario.csv),
    entrena modelos por variable, genera forecast diario a 1 año,
    lo exporta a CSV, lo sube por FTP y dispara importación PHP.
    """
    data_path = os.getenv("DATA_PATH", "data/clima_diario.csv")
    if not os.path.exists(data_path):
        raise RuntimeError(f"No existe el dataset diario en: {data_path}")

    df = pd.read_csv(data_path)
    # Validación mínima
    expected = {"fecha", "temp_media", "hum_media", "rad_media", "pres_media", "precip_total", "viento_media"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV diario incompleto. Faltan columnas: {sorted(missing)}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    # Asegurar numéricos
    for c in ["temp_media", "hum_media", "rad_media", "pres_media", "precip_total", "viento_media"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Predicciones 365 días (diarias)
# Predicciones y renombre de yhat ANTES del merge
    temp = fit_predict(df, "temp_media").rename(columns={"yhat": "temp_media"})
    hum  = fit_predict(df, "hum_media", cap99=True).rename(columns={"yhat": "hum_media"})
    rad  = fit_predict(df, "rad_media", non_negative=True).rename(columns={"yhat": "rad_media"})
    pres = fit_predict(df, "pres_media").rename(columns={"yhat": "pres_media"})
    prec = fit_predict(df, "precip_total", non_negative=True, log1p=True).rename(columns={"yhat": "precip_total"})
    wind = fit_predict(df, "viento_media", non_negative=True).rename(columns={"yhat": "viento_media"})

    # Merge seguro (sin columnas duplicadas)
    out = temp.merge(hum, on="fecha", how="inner")
    out = out.merge(rad, on="fecha", how="inner")
    out = out.merge(pres, on="fecha", how="inner")
    out = out.merge(prec, on="fecha", how="inner")
    out = out.merge(wind, on="fecha", how="inner")


    trained_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Metadata para que tu PHP lo guarde en la BD
    out["modelo_version"] = MODEL_VERSION
    out["entrenado_en"] = trained_at

    # Generar CSV
    csv_path = "forecast_1y.csv"
    out.to_csv(csv_path, index=False)

    # Subir por FTP
    upload_csv_ftp(csv_path)

    # Importar en hosting (PHP -> MySQL local)
    php_result = trigger_php_import()

    return {
        "inserted_rows": int(len(out)),
        "trained_at_utc": trained_at,
        "csv_generated": csv_path,
        "php_import_result": php_result[:2000],
    }
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Evitar división por 0 en MAPE
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else None

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(float(mape), 4) if mape is not None else None,
    }

def evaluate_variable(df, col, cap99=False, non_negative=False, log1p=False, test_ratio=0.2):
    data = df[["fecha", col]].dropna().copy()
    data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
    data = data.dropna(subset=["fecha"]).sort_values("fecha")
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[col])

    n = len(data)
    if n < 60:
        raise RuntimeError(f"No hay suficientes datos para evaluar {col} (mínimo ~60 días).")

    split = int(n * (1 - test_ratio))
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    train_p = train.rename(columns={"fecha": "ds", col: "y"})[["ds", "y"]].copy()
    test_p  = test.rename(columns={"fecha": "ds", col: "y"})[["ds", "y"]].copy()

    if log1p:
        train_p["y"] = np.log1p(np.maximum(train_p["y"].astype(float), 0))

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train_p)

    future = m.make_future_dataframe(periods=len(test_p), freq="D")
    fc = m.predict(future).tail(len(test_p))[["ds", "yhat"]].copy()

    y_pred = fc["yhat"].to_numpy()
    if log1p:
        y_pred = np.expm1(y_pred)
    if non_negative:
        y_pred = np.maximum(y_pred, 0)
    if cap99:
        y_pred = np.minimum(y_pred, 99.99)

    y_true = test_p["y"].to_numpy(dtype=float)
    # si hiciste log1p en train, y_true sigue en escala normal (correcto)

    return compute_metrics(y_true, y_pred), {
        "test_days": int(len(test_p)),
        "train_days": int(len(train_p)),
        "test_start": str(test_p["ds"].min().date()),
        "test_end": str(test_p["ds"].max().date()),
    }

def get_model_metrics():
    data_path = os.getenv("DATA_PATH", "data/clima_diario.csv")
    df = pd.read_csv(data_path)

    # Métricas por variable
    metrics = {}

    m_temp, info = evaluate_variable(df, "temp_media")
    metrics["temp_media"] = {**m_temp, **info}

    m_hum, info = evaluate_variable(df, "hum_media", cap99=True)
    metrics["hum_media"] = {**m_hum, **info}

    m_rad, info = evaluate_variable(df, "rad_media", non_negative=True)
    metrics["rad_media"] = {**m_rad, **info}

    m_pres, info = evaluate_variable(df, "pres_media")
    metrics["pres_media"] = {**m_pres, **info}

    m_prec, info = evaluate_variable(df, "precip_total", non_negative=True, log1p=True)
    metrics["precip_total"] = {**m_prec, **info}

    m_wind, info = evaluate_variable(df, "viento_media", non_negative=True)
    metrics["viento_media"] = {**m_wind, **info}

    return {
        "model_version": MODEL_VERSION,
        "evaluated_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
