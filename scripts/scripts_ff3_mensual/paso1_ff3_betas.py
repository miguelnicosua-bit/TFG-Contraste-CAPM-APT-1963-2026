"""
PASO 1 FF3 - Primera etapa: estimación de las betas del modelo de 3 factores
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Para cada una de las 100 carteras estimamos por MCO la regresión:

    r_{j,t} - r_{f,t} = alpha_j + beta_j^MKT (r_M - r_f)_t
                                 + beta_j^SMB SMB_t
                                 + beta_j^HML HML_t
                                 + e_{j,t}

Guardamos para cada cartera:
  - alpha_j (debe ser nulo bajo FF3)
  - tres betas (MKT, SMB, HML)
  - errores estándar, t-estadísticos
  - R^2 de cada regresión (debe ser mayor que en el CAPM)
  - desviación típica del residuo
  - rendimiento medio en exceso
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ========================================================================
# CARGA DE DATOS PROCESADOS
# ========================================================================
CARPETA_PROCESADOS = "datos_procesados"
CARPETA_RESULTADOS = "resultados/tablas/ff3_mensual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
factores_ff3  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))

print(f"Datos cargados: {excesos.shape[0]} meses x {excesos.shape[1]} carteras")
print(f"Factores: {list(factores_ff3.columns)}")

# ========================================================================
# REGRESIÓN DE SERIE TEMPORAL POR CARTERA (3 factores)
# ========================================================================
# Para cada cartera, regresamos su exceso de rendimiento sobre los 3 factores.
# La constante captura alpha_j; los 3 coeficientes son las betas factoriales.

resultados = []

for nombre_cartera in excesos.columns:
    y = excesos[nombre_cartera]

    # Tratamiento de NaN
    datos = pd.concat([y, factores_ff3], axis=1).dropna()
    y_clean = datos.iloc[:, 0]
    X_clean = datos.iloc[:, 1:]

    # Añadimos constante y estimamos por MCO
    X = sm.add_constant(X_clean)
    modelo = sm.OLS(y_clean, X).fit()

    resultados.append({
        "cartera":         nombre_cartera,
        "alpha":           modelo.params["const"],
        "beta_MKT":        modelo.params["Mkt-RF"],
        "beta_SMB":        modelo.params["SMB"],
        "beta_HML":        modelo.params["HML"],
        "alpha_se":        modelo.bse["const"],
        "alpha_t":         modelo.tvalues["const"],
        "beta_MKT_t":      modelo.tvalues["Mkt-RF"],
        "beta_SMB_t":      modelo.tvalues["SMB"],
        "beta_HML_t":      modelo.tvalues["HML"],
        "r2":              modelo.rsquared,
        "r2_adj":          modelo.rsquared_adj,
        "sigma_residual":  np.sqrt(modelo.mse_resid),
        "rend_medio":      y_clean.mean(),
        "n_obs":           int(modelo.nobs),
    })

res = pd.DataFrame(resultados).set_index("cartera")

# ========================================================================
# RESUMEN: ESTADÍSTICOS DESCRIPTIVOS
# ========================================================================
print("\n" + "=" * 70)
print("ESTADÍSTICOS DESCRIPTIVOS DE LAS 100 REGRESIONES FF3")
print("=" * 70)

resumen = pd.DataFrame({
    "Media":    [res["beta_MKT"].mean(), res["beta_SMB"].mean(), res["beta_HML"].mean()],
    "Mediana":  [res["beta_MKT"].median(), res["beta_SMB"].median(), res["beta_HML"].median()],
    "Mínimo":   [res["beta_MKT"].min(), res["beta_SMB"].min(), res["beta_HML"].min()],
    "Máximo":   [res["beta_MKT"].max(), res["beta_SMB"].max(), res["beta_HML"].max()],
    "Desv tip": [res["beta_MKT"].std(), res["beta_SMB"].std(), res["beta_HML"].std()],
}, index=["beta_MKT", "beta_SMB", "beta_HML"])
print(resumen.round(3))

print(f"\nR^2 medio (FF3):     {res['r2'].mean():.4f}")
print(f"R^2 medio ajustado:  {res['r2_adj'].mean():.4f}")

# ========================================================================
# ALFAS DE JENSEN: ¿se reducen respecto al CAPM?
# ========================================================================
print("\n" + "=" * 70)
print("ALFAS DE JENSEN BAJO FF3")
print("=" * 70)
print(f"Alpha medio anualizado (%):  {res['alpha'].mean() * 12 * 100:.3f}")
print(f"Alfas significativos al 5%:  {(res['alpha_t'].abs() > 1.96).sum()} / 100")
print(f"Alfas significativos al 1%:  {(res['alpha_t'].abs() > 2.58).sum()} / 100")
print(f"|alpha| medio (anualizado):  {res['alpha'].abs().mean() * 12 * 100:.3f}%")

# ========================================================================
# COMPARACIÓN DIRECTA CON LOS RESULTADOS DEL CAPM
# ========================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN CAPM vs FF3 (PRIMERA ETAPA)")
print("=" * 70)

# Cargamos los resultados del CAPM
capm = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))

comparacion = pd.DataFrame({
    "Métrica": [
        "R^2 medio",
        "R^2 mediano",
        "|alpha| medio anual (%)",
        "Alfas significativos al 5% (de 100)",
        "Alfas significativos al 1% (de 100)",
    ],
    "CAPM": [
        f"{capm['r2'].mean():.4f}",
        f"{capm['r2'].median():.4f}",
        f"{capm['alpha'].abs().mean() * 12 * 100:.3f}",
        f"{(capm['alpha_t'].abs() > 1.96).sum()}",
        f"{(capm['alpha_t'].abs() > 2.58).sum()}",
    ],
    "FF3": [
        f"{res['r2'].mean():.4f}",
        f"{res['r2'].median():.4f}",
        f"{res['alpha'].abs().mean() * 12 * 100:.3f}",
        f"{(res['alpha_t'].abs() > 1.96).sum()}",
        f"{(res['alpha_t'].abs() > 2.58).sum()}",
    ],
})
print(comparacion.to_string(index=False))

# ========================================================================
# TOP 5 CARTERAS CON MAYOR Y MENOR ALFA (FF3)
# ========================================================================
print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MAYOR ALFA BAJO FF3 (anualizado)")
print("=" * 70)
top5 = res.nlargest(5, "alpha")[["alpha", "alpha_t", "beta_MKT", "beta_SMB", "beta_HML", "r2"]].copy()
top5["alpha"] = top5["alpha"] * 12 * 100
print(top5.round(3))

print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MENOR ALFA BAJO FF3 (anualizado)")
print("=" * 70)
bot5 = res.nsmallest(5, "alpha")[["alpha", "alpha_t", "beta_MKT", "beta_SMB", "beta_HML", "r2"]].copy()
bot5["alpha"] = bot5["alpha"] * 12 * 100
print(bot5.round(3))

# ========================================================================
# GUARDADO
# ========================================================================
res.to_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))
res.to_csv(os.path.join(CARPETA_RESULTADOS, "primera_etapa_ff3.csv"))

print(f"\n✓ Resultados guardados en '{CARPETA_PROCESADOS}/primera_etapa_ff3.pkl'")
print("✓ Listos para ejecutar paso2_ff3_seccion_cruzada.py")
