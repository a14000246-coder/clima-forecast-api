import pandas as pd
import numpy as np
from datetime import datetime
import mysql.connector
from prophet import Prophet

from .settings import DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME, MODEL_VERSION, HORIZON_DAYS

SQL_DAILY = """
SELECT
  DATE(fecha) AS fecha,
  AVG(temperatura) AS temp_media,
  AVG(humedad_relativa) AS hum_media,
  AVG(radiacion_solar) AS rad_media,
  AVG(presion_atmosferica) AS pres_media,
  SUM(precipitacion) AS precip_total,
  AVG(velocidad_viento) AS viento_media
FROM clima
GROUP BY DATE(fecha)
ORDER BY fecha;
"""

def db_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS, database=DB_NAME
    )

def fit_predict(df, col, non_negative=False, cap99=False, log1p=False):
    data = df[["fecha", col]].dropna().copy()
    data.rename(columns={"fecha":"ds", col:"y"}, inplace=True)

    if log1p:
        data["y"] = np.log1p(np.maximum(data["y"].astype(float), 0))

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(data)

    future = m.make_future_dataframe(periods=HORIZON_DAYS, freq="D")
    fc = m.predict(future)[["ds","yhat"]].tail(HORIZON_DAYS).copy()

    yhat = fc["yhat"].to_numpy()
    if log1p:
        yhat = np.expm1(yhat)
    if non_negative:
        yhat = np.maximum(yhat, 0)
    if cap99:
        yhat = np.minimum(yhat, 99.99)

    fc["fecha"] = pd.to_datetime(fc["ds"]).dt.date
    fc["yhat"] = np.round(yhat, 2)
    return fc[["fecha","yhat"]]

def run_forecast():
    conn = db_conn()
    df = pd.read_sql(SQL_DAILY, conn)
    df["fecha"] = pd.to_datetime(df["fecha"])

    temp = fit_predict(df, "temp_media")
    hum  = fit_predict(df, "hum_media", cap99=True)
    rad  = fit_predict(df, "rad_media", non_negative=True)
    pres = fit_predict(df, "pres_media")
    prec = fit_predict(df, "precip_total", non_negative=True, log1p=True)
    wind = fit_predict(df, "viento_media", non_negative=True)

    out = temp.merge(hum, on="fecha", suffixes=("_temp","_hum"))
    out = out.merge(rad, on="fecha")
    out = out.merge(pres, on="fecha")
    out = out.merge(prec, on="fecha")
    out = out.merge(wind, on="fecha")

    out.columns = ["fecha","temp_media","hum_media","rad_media","pres_media","precip_total","viento_media"]

    trained_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()

    cur.execute("DELETE FROM clima_forecast_diario WHERE fecha >= CURDATE();")

    ins = """
    INSERT INTO clima_forecast_diario
    (fecha, temp_media, hum_media, rad_media, pres_media, precip_total, viento_media, modelo_version, entrenado_en)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
      temp_media=VALUES(temp_media),
      hum_media=VALUES(hum_media),
      rad_media=VALUES(rad_media),
      pres_media=VALUES(pres_media),
      precip_total=VALUES(precip_total),
      viento_media=VALUES(viento_media),
      modelo_version=VALUES(modelo_version),
      entrenado_en=VALUES(entrenado_en);
    """

    rows = []
    for _, r in out.iterrows():
        rows.append((
            r["fecha"], float(r["temp_media"]), float(r["hum_media"]),
            float(r["rad_media"]), float(r["pres_media"]),
            float(r["precip_total"]), float(r["viento_media"]),
            MODEL_VERSION, trained_at
        ))

    cur.executemany(ins, rows)
    conn.commit()
    cur.close()
    conn.close()

    return {"inserted_rows": len(rows), "trained_at_utc": trained_at}
