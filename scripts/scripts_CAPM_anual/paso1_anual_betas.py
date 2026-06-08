"""
PASO 1 ANUAL - Primera etapa CAPM con datos anuales
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Para cada cartera j, estimamos por MCO la regresión del modelo de mercado
sobre la serie temporal anual:

    r_{j,t} - r_{f,t} = alpha_j + beta_j (r_{M,t} - r_{f,t}) + e_{j,t}

donde t indexa años (no meses).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

CARPETA_ANUAL = "datos_procesados_anual"
CARPETA_RESULTADOS = "resultados/tablas/capm_anual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos = pd.read_pickle(os.path.join(CARPETA_ANUAL, "excesos_carteras_anual.pkl"))
mkt_rf  = pd.read_pickle(os.path.join(CARPETA_ANUAL, "mkt_rf_anual.pkl"))

print(f"Datos anuales cargados: {excesos.shape[0]} años x {excesos.shape[1]} carteras")

# ========================================================================
# REGRESIÓN DE SERIE TEMPORAL POR CARTERA
# ========================================================================
resultados = []

for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, mkt_rf.rename("Mkt-RF")], axis=1).dropna()

    X = sm.add_constant(datos["Mkt-RF"])
    modelo = sm.OLS(datos.iloc[:, 0], X).fit()

    resultados.append({
        "cartera":        cartera,
        "alpha":          modelo.params["const"],
        "beta":           modelo.params["Mkt-RF"],
        "alpha_se":       modelo.bse["const"],
        "alpha_t":        modelo.tvalues["const"],
        "beta_t":         modelo.tvalues["Mkt-RF"],
        "r2":             modelo.rsquared,
        "sigma_residual": np.sqrt(modelo.mse_resid),
        "rend_medio":     datos.iloc[:, 0].mean(),
        "n_obs":          int(modelo.nobs),
    })

res = pd.DataFrame(resultados).set_index("cartera")

# ========================================================================
# ESTADÍSTICOS DESCRIPTIVOS
# ========================================================================
print("\n" + "=" * 70)
print("ESTADÍSTICOS DESCRIPTIVOS DE LAS 100 REGRESIONES ANUALES")
print("=" * 70)
print(f"Beta media:    {res['beta'].mean():.3f}")
print(f"Beta mediana:  {res['beta'].median():.3f}")
print(f"Beta mínima:   {res['beta'].min():.3f}  ({res['beta'].idxmin()})")
print(f"Beta máxima:   {res['beta'].max():.3f}  ({res['beta'].idxmax()})")
print(f"Desv tip:      {res['beta'].std():.3f}")

print(f"\nR² medio:      {res['r2'].mean():.4f}")
print(f"R² mediano:    {res['r2'].median():.4f}")
print(f"R² mínimo:     {res['r2'].min():.4f}  ({res['r2'].idxmin()})")
print(f"R² máximo:     {res['r2'].max():.4f}  ({res['r2'].idxmax()})")

print(f"\nAlpha medio anual (%):                {res['alpha'].mean() * 100:.3f}")
print(f"|Alpha| medio anual (%):              {res['alpha'].abs().mean() * 100:.3f}")
print(f"Alfas significativos al 5% (|t|>1.96): {(res['alpha_t'].abs() > 1.96).sum()} / 100")
print(f"Alfas significativos al 1% (|t|>2.58): {(res['alpha_t'].abs() > 2.58).sum()} / 100")

# ========================================================================
# COMPARACIÓN CON LOS RESULTADOS MENSUALES
# ========================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN PRIMERA ETAPA: MENSUAL vs ANUAL")
print("=" * 70)

CARPETA_PROCESADOS = "datos_procesados"
mensual = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))

comparacion = pd.DataFrame({
    "Métrica": [
        "Beta media",
        "R² medio",
        "|alpha| medio anual (%)",
        "Alfas significativos al 5% (de 100)",
    ],
    "Mensual": [
        f"{mensual['beta'].mean():.3f}",
        f"{mensual['r2'].mean():.4f}",
        f"{mensual['alpha'].abs().mean() * 12 * 100:.3f}",
        f"{(mensual['alpha_t'].abs() > 1.96).sum()}",
    ],
    "Anual": [
        f"{res['beta'].mean():.3f}",
        f"{res['r2'].mean():.4f}",
        f"{res['alpha'].abs().mean() * 100:.3f}",
        f"{(res['alpha_t'].abs() > 1.96).sum()}",
    ],
})
print(comparacion.to_string(index=False))

# ========================================================================
# TOP 5 CARTERAS CON ALFAS EXTREMOS
# ========================================================================
print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MAYOR ALFA ANUAL")
print("=" * 70)
top5 = res.nlargest(5, "alpha")[["alpha", "alpha_t", "beta", "r2"]].copy()
top5["alpha"] = top5["alpha"] * 100
print(top5.round(3))

print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MENOR ALFA ANUAL")
print("=" * 70)
bot5 = res.nsmallest(5, "alpha")[["alpha", "alpha_t", "beta", "r2"]].copy()
bot5["alpha"] = bot5["alpha"] * 100
print(bot5.round(3))

# ========================================================================
# GUARDADO
# ========================================================================
res.to_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_anual.pkl"))
res.to_csv(os.path.join(CARPETA_RESULTADOS, "primera_etapa_anual.csv"))

print(f"\n✓ Resultados guardados en '{CARPETA_ANUAL}/primera_etapa_anual.pkl'")
print("✓ Listos para ejecutar paso2_anual_seccion_cruzada.py")
