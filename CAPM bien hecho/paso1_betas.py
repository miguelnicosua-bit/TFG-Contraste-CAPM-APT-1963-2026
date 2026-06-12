"""
PASO 1 - Primera etapa de Fama-MacBeth: estimación de betas
TFG Valoración de Activos - Contraste empírico del CAPM
Autor: Miguel Suárez Crespo

Implementación de la primera etapa de Fama y MacBeth (1973) según la
formulación del paper original (Marín y Rubio, 2011, cap. 11). Para cada
cartera j se estima por MCO la regresión del modelo de mercado:

    r_{j,t} - r_{f,t} = alpha_j + beta_j (r_{M,t} - r_{f,t}) + e_{j,t}

Se generan DOS estimaciones diferentes:

  (A) BETAS ESTÁTICAS (toda la muestra del TFG, T=754):
      Sirven exclusivamente como descriptivos: Tabla 1, mapas de calor de
      betas y alfas, distribución de t-estadísticos (Figuras 2-5 del TFG).

  (B) BETAS ROLLING (ventana móvil de 60 meses, T=754 efectivo):
      Son las que se usan en la SEGUNDA ETAPA. Para CADA mes t del periodo
      muestral (julio 1963 - abril 2026), las 100 betas de la sección cruzada
      del mes t se estiman con los 60 meses previos [t-60, t-1] usando los
      DATOS EXTENDIDOS (desde julio 1958). De este modo:
        - Para julio 1963 (primer mes), las betas se estiman con
          datos de julio 1958 - junio 1963.
        - Para febrero 1963 (sic, "1963" mantenido por coherencia con la
          indicación del tutor): se descarta enero 1958, se añade enero 1963.
        - Y así sucesivamente hasta abril 2026.
      La segunda etapa tiene por tanto T = 754 regresiones transversales
      (todo el periodo muestral del TFG, sin pérdida de observaciones).

Productos del paso 1:
  - primera_etapa.pkl              -> Estáticas (descriptivos)
  - primera_etapa_rolling.pkl      -> Diccionario con tres DataFrames T x N:
                                       betas_rolling, sigma_resid_rolling,
                                       alpha_rolling. Dimensión 754 x 100.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ========================================================================
# CARGA DE DATOS PROCESADOS DEL PASO 0
# ========================================================================
CARPETA_PROCESADOS = "datos_procesados"
CARPETA_RESULTADOS = "resultados/tablas/capm_mensual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# Versión principal (T=754, julio 1963 - abril 2026)
excesos = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))

# Versión extendida con 60 meses previos (julio 1958 - abril 2026)
excesos_ext = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_extendido.pkl"))
mkt_rf_ext  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf_extendido.pkl"))

print(f"Datos cargados:")
print(f"  Periodo muestral (T): {excesos.shape[0]} meses x {excesos.shape[1]} carteras")
print(f"  Periodo extendido:    {excesos_ext.shape[0]} meses (incluye 60 meses previos)")

# Parámetro de la ventana móvil (Fama y MacBeth, 1973)
VENTANA = 60

# ========================================================================
# (A) BETAS ESTÁTICAS - PARA DESCRIPTIVOS Y MAPAS DE CALOR
# ========================================================================
# Para cada cartera j: MCO con toda la muestra del periodo muestral del TFG
# (julio 1963 - abril 2026, T=754). Sirven para la Tabla 1 (descriptivos)
# y los mapas de calor de la primera etapa que ya estaban en el TFG.

resultados = []

for nombre_cartera in excesos.columns:
    y = excesos[nombre_cartera]
    datos = pd.concat([y, mkt_rf], axis=1).dropna()
    y_clean = datos.iloc[:, 0]
    x_clean = datos.iloc[:, 1]
    X = sm.add_constant(x_clean)
    modelo = sm.OLS(y_clean, X).fit()

    resultados.append({
        "cartera": nombre_cartera,
        "alpha":          modelo.params["const"],
        "beta":           modelo.params["Mkt-RF"],
        "alpha_se":       modelo.bse["const"],
        "beta_se":        modelo.bse["Mkt-RF"],
        "alpha_t":        modelo.tvalues["const"],
        "beta_t":         modelo.tvalues["Mkt-RF"],
        "r2":             modelo.rsquared,
        "sigma_residual": np.sqrt(modelo.mse_resid),
        "rend_medio":     y_clean.mean(),
        "n_obs":          int(modelo.nobs),
    })

res = pd.DataFrame(resultados).set_index("cartera")

# ========================================================================
# RESUMEN: DISTRIBUCIÓN DE LAS BETAS Y ALFAS ESTÁTICOS
# ========================================================================
print("\n" + "=" * 70)
print("(A) BETAS ESTÁTICAS - DISTRIBUCIÓN DE LAS 100 BETAS ESTIMADAS")
print("=" * 70)
print(f"Beta media:    {res['beta'].mean():.3f}")
print(f"Beta mediana:  {res['beta'].median():.3f}")
print(f"Beta mínima:   {res['beta'].min():.3f}  ({res['beta'].idxmin()})")
print(f"Beta máxima:   {res['beta'].max():.3f}  ({res['beta'].idxmax()})")
print(f"Desv. típica:  {res['beta'].std():.3f}")

print("\n" + "=" * 70)
print("ALFAS DE JENSEN (anualizados, en %)")
print("=" * 70)
print(f"Alpha medio (anualizado):  {res['alpha'].mean() * 12 * 100:.3f}%")
print(f"Alfas significativos al 5% (|t|>1.96): "
      f"{(res['alpha_t'].abs() > 1.96).sum()} de 100")
print(f"Alfas significativos al 1% (|t|>2.58): "
      f"{(res['alpha_t'].abs() > 2.58).sum()} de 100")

print("\n" + "=" * 70)
print("R^2 DE LAS REGRESIONES DE SERIE TEMPORAL")
print("=" * 70)
print(f"R^2 medio:    {res['r2'].mean():.3f}")
print(f"R^2 mediano:  {res['r2'].median():.3f}")
print(f"R^2 mínimo:   {res['r2'].min():.3f}  ({res['r2'].idxmin()})")
print(f"R^2 máximo:   {res['r2'].max():.3f}  ({res['r2'].idxmax()})")

# 5 CARTERAS CON MAYOR Y MENOR ALFA (Jensen)
print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MAYOR ALFA (anualizado)")
print("=" * 70)
top5 = res.nlargest(5, "alpha")[["alpha", "alpha_t", "beta", "r2"]].copy()
top5["alpha"] = top5["alpha"] * 12 * 100
print(top5.round(3))

print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MENOR ALFA (anualizado)")
print("=" * 70)
bot5 = res.nsmallest(5, "alpha")[["alpha", "alpha_t", "beta", "r2"]].copy()
bot5["alpha"] = bot5["alpha"] * 12 * 100
print(bot5.round(3))

# ========================================================================
# (B) BETAS ROLLING - METODOLOGÍA FAMA Y MACBETH (1973)
# ========================================================================
# Para cada mes t del periodo muestral (julio 1963 - abril 2026):
#   Para cada cartera j:
#     Estimar (alpha_j, beta_j) por MCO sobre los 60 meses previos al mes t,
#     usando los datos EXTENDIDOS (que incluyen julio 1958 - junio 1963).
#   Las betas estimadas se usarán en la regresión transversal del mes t
#   (paso 2). La beta es, por tanto, una función del tiempo: beta_{j,t}.
#
# El primer mes del periodo muestral (julio 1963) usa la ventana
# [julio 1958, junio 1963] disponible en el dataset extendido.
# El último mes (abril 2026) usa la ventana [mayo 2021, abril 2026].
# Por tanto el panel de betas rolling tiene exactamente T = 754 filas.

print("\n" + "=" * 70)
print(f"(B) BETAS ROLLING - Ventana móvil de {VENTANA} meses")
print("=" * 70)
print(f"Fama y MacBeth (1973): para cada mes t del periodo muestral, se")
print(f"re-estiman las 100 betas con los {VENTANA} meses anteriores. Esas")
print(f"betas se emplean en la regresión transversal del mes t (paso 2).")
print(f"\nFechas extremas:")
print(f"  Primer mes del muestral:  {excesos.index[0]}  "
      f"-> ventana [{excesos_ext.index[0]}, {excesos.index[0] - 1}]")
print(f"  Último mes del muestral:  {excesos.index[-1]}  "
      f"-> ventana [{excesos.index[-1] - VENTANA}, {excesos.index[-1] - 1}]")
print(f"\nTotal de secciones cruzadas: {len(excesos)} (todo el periodo muestral).")

# Estructura de salida: DataFrames T x N indexados por el periodo muestral
fechas_seccion_cruzada = excesos.index  # las 754 fechas del muestral

betas_rolling        = pd.DataFrame(index=fechas_seccion_cruzada, columns=excesos.columns, dtype=float)
alpha_rolling        = pd.DataFrame(index=fechas_seccion_cruzada, columns=excesos.columns, dtype=float)
sigma_resid_rolling  = pd.DataFrame(index=fechas_seccion_cruzada, columns=excesos.columns, dtype=float)

print(f"\nEstimando betas rolling para cada cartera y cada mes...")
print(f"({len(fechas_seccion_cruzada)} ventanas x {excesos.shape[1]} carteras = "
      f"{len(fechas_seccion_cruzada) * excesos.shape[1]:,} regresiones)")

# Para cada mes t del muestral, encontrar su posición en el extendido y
# tomar la ventana [t-60, t-1] en el extendido.
for fecha_t in fechas_seccion_cruzada:
    # Posición de fecha_t en el dataset extendido
    pos_ext = excesos_ext.index.get_loc(fecha_t)
    # Ventana: 60 meses anteriores (no incluye fecha_t)
    idx_inicio = pos_ext - VENTANA
    idx_fin    = pos_ext              # excluido del slice
    fechas_ventana = excesos_ext.index[idx_inicio:idx_fin]

    # Datos del mercado en la ventana
    mkt_ventana = mkt_rf_ext.loc[fechas_ventana]
    X_ventana   = sm.add_constant(mkt_ventana)

    # Estimación cartera por cartera
    for cartera in excesos_ext.columns:
        y_ventana = excesos_ext.loc[fechas_ventana, cartera]
        datos = pd.concat([y_ventana, X_ventana], axis=1).dropna()

        if len(datos) < 30:
            continue

        modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
        alpha_rolling.loc[fecha_t, cartera]       = modelo.params["const"]
        betas_rolling.loc[fecha_t, cartera]       = modelo.params["Mkt-RF"]
        sigma_resid_rolling.loc[fecha_t, cartera] = np.sqrt(modelo.mse_resid)

# Resumen de las betas rolling
print("\n" + "-" * 70)
print(f"Betas rolling estimadas: {betas_rolling.shape[0]} meses x {betas_rolling.shape[1]} carteras")
print(f"Datos faltantes en betas rolling: {betas_rolling.isna().sum().sum()}")
print(f"Beta media (promedio entre todas las celdas): {betas_rolling.stack().mean():.3f}")
print(f"Beta mediana:                                  {betas_rolling.stack().median():.3f}")

# ========================================================================
# GUARDADO PARA LOS SIGUIENTES PASOS
# ========================================================================
# (A) Betas estáticas (para descriptivos y mapas de calor)
res.to_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))
res.to_csv(os.path.join(CARPETA_RESULTADOS, "primera_etapa.csv"))

# (B) Betas rolling (para la segunda etapa, paso 2)
primera_rolling = {
    "betas_rolling":        betas_rolling,
    "alpha_rolling":        alpha_rolling,
    "sigma_resid_rolling":  sigma_resid_rolling,
    "ventana":              VENTANA,
}
pd.to_pickle(primera_rolling, os.path.join(CARPETA_PROCESADOS, "primera_etapa_rolling.pkl"))

print(f"\n✓ (A) Betas estáticas guardadas en '{CARPETA_PROCESADOS}/primera_etapa.pkl'")
print(f"✓ (B) Betas rolling guardadas en '{CARPETA_PROCESADOS}/primera_etapa_rolling.pkl'")
print(f"✓ Tabla CSV exportada a '{CARPETA_RESULTADOS}/primera_etapa.csv'")
print("✓ Listos para ejecutar paso2_seccion_cruzada.py")
