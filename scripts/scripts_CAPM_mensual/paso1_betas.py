"""
PASO 1 - Primera etapa de Fama-MacBeth: estimación de las 100 betas
TFG Valoración de Activos - Contraste empírico del CAPM

Para cada una de las 100 carteras estimamos por MCO la regresión:

    r_{i,t} - r_{f,t} = alpha_i + beta_i (r_{M,t} - r_{f,t}) + e_{i,t}

Guardamos:
  - alpha_i, beta_i (estimaciones puntuales)
  - errores estándar y t-estadísticos
  - R^2 de cada regresión
  - desviación típica del residuo (riesgo idiosincrático)
  - rendimiento medio en exceso de cada cartera (para la segunda etapa)
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

excesos = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))

print(f"Datos cargados: {excesos.shape[0]} meses x {excesos.shape[1]} carteras")

# ========================================================================
# REGRESIÓN DE SERIE TEMPORAL PARA CADA CARTERA
# ========================================================================
# Para cada cartera i:
#   y = excesos[i]  (exceso de rendimiento de la cartera)
#   X = [1, Mkt-RF]  (constante + factor de mercado)
# La constante captura alpha_i; el coeficiente de Mkt-RF captura beta_i.

resultados = []

for nombre_cartera in excesos.columns:
    y = excesos[nombre_cartera]

    # Tratamiento de NaN: dropna alinea y con X eliminando los meses
    # donde la cartera tenga dato faltante. Cada cartera usa solo sus
    # observaciones válidas (es lo que hace cualquier paper estándar).
    datos = pd.concat([y, mkt_rf], axis=1).dropna()
    y_clean = datos.iloc[:, 0]
    x_clean = datos.iloc[:, 1]

    # Añadimos la constante (statsmodels no la incluye por defecto)
    X = sm.add_constant(x_clean)

    # Regresión MCO
    modelo = sm.OLS(y_clean, X).fit()

    resultados.append({
        "cartera": nombre_cartera,
        "alpha": modelo.params["const"],
        "beta":  modelo.params["Mkt-RF"],
        "alpha_se": modelo.bse["const"],
        "beta_se":  modelo.bse["Mkt-RF"],
        "alpha_t":  modelo.tvalues["const"],
        "beta_t":   modelo.tvalues["Mkt-RF"],
        "r2":       modelo.rsquared,
        "sigma_residual": np.sqrt(modelo.mse_resid),  # vol del riesgo idiosincrático
        "rend_medio": y_clean.mean(),                  # para la segunda etapa
        "n_obs": int(modelo.nobs),
    })

# Convertimos a DataFrame y usamos el nombre de cartera como índice
res = pd.DataFrame(resultados).set_index("cartera")

# ========================================================================
# RESUMEN: DISTRIBUCIÓN DE LAS BETAS Y ALFAS
# ========================================================================
print("\n" + "=" * 70)
print("DISTRIBUCIÓN DE LAS 100 BETAS ESTIMADAS")
print("=" * 70)
print(f"Beta media:    {res['beta'].mean():.3f}")
print(f"Beta mediana:  {res['beta'].median():.3f}")
print(f"Beta mínima:   {res['beta'].min():.3f}  ({res['beta'].idxmin()})")
print(f"Beta máxima:   {res['beta'].max():.3f}  ({res['beta'].idxmax()})")
print(f"Desv. típica:  {res['beta'].std():.3f}")

print("\n" + "=" * 70)
print("ALFAS DE JENSEN (anualizados, en %)")
print("=" * 70)
# Alfa anualizado = alpha mensual * 12 * 100
print(f"Alpha medio (anualizado):  {res['alpha'].mean() * 12 * 100:.3f}%")
print(f"Alfas significativos al 5% (|t|>1.96): "
      f"{(res['alpha_t'].abs() > 1.96).sum()} de 100")
print(f"Alfas significativos al 1% (|t|>2.58): "
      f"{(res['alpha_t'].abs() > 2.58).sum()} de 100")

# El CAPM predice que todos los alfas deben ser CERO. Si encontramos muchos
# significativamente distintos de cero, es evidencia EN CONTRA del CAPM.

print("\n" + "=" * 70)
print("R^2 DE LAS REGRESIONES DE SERIE TEMPORAL")
print("=" * 70)
print(f"R^2 medio:    {res['r2'].mean():.3f}")
print(f"R^2 mediano:  {res['r2'].median():.3f}")
print(f"R^2 mínimo:   {res['r2'].min():.3f}  ({res['r2'].idxmin()})")
print(f"R^2 máximo:   {res['r2'].max():.3f}  ({res['r2'].idxmax()})")

# ========================================================================
# 5 CARTERAS CON MAYOR Y MENOR ALFA (Jensen)
# ========================================================================
print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MAYOR ALFA (anualizado)")
print("=" * 70)
top5 = res.nlargest(5, "alpha")[["alpha", "alpha_t", "beta", "r2"]].copy()
top5["alpha"] = top5["alpha"] * 12 * 100  # anualizar y a %
print(top5.round(3))

print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MENOR ALFA (anualizado)")
print("=" * 70)
bot5 = res.nsmallest(5, "alpha")[["alpha", "alpha_t", "beta", "r2"]].copy()
bot5["alpha"] = bot5["alpha"] * 12 * 100
print(bot5.round(3))

# ========================================================================
# GUARDADO PARA LOS SIGUIENTES PASOS
# ========================================================================
res.to_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))
res.to_csv(os.path.join(CARPETA_RESULTADOS, "primera_etapa.csv"))

print(f"\n✓ Resultados guardados en '{CARPETA_PROCESADOS}/primera_etapa.pkl'")
print(f"✓ Tabla CSV exportada a '{CARPETA_RESULTADOS}/primera_etapa.csv'")
print("✓ Listos para ejecutar paso2_seccion_cruzada.py")
