"""
PASO 1 FF3 ANUAL - Primera etapa del modelo de tres factores con datos anuales
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Para cada cartera j, estimamos por MCO:

    r_{j,t} - r_{f,t} = alpha_j + beta_j^MKT (r_M - r_f)_t
                                 + beta_j^SMB SMB_t
                                 + beta_j^HML HML_t
                                 + e_{j,t}

donde t indexa años (no meses).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

CARPETA_ANUAL = "datos_procesados_anual"
CARPETA_RESULTADOS = "resultados/tablas/ff3_anual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos        = pd.read_pickle(os.path.join(CARPETA_ANUAL, "excesos_carteras_anual.pkl"))
factores_ff3   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "factores_ff3_anual.pkl"))

print(f"Datos anuales: {excesos.shape[0]} años x {excesos.shape[1]} carteras")
print(f"Factores: {list(factores_ff3.columns)}")

# ========================================================================
# REGRESIÓN DE SERIE TEMPORAL POR CARTERA (3 factores, anuales)
# ========================================================================
resultados = []

for nombre_cartera in excesos.columns:
    y = excesos[nombre_cartera]
    datos = pd.concat([y, factores_ff3], axis=1).dropna()
    y_clean = datos.iloc[:, 0]
    X_clean = datos.iloc[:, 1:]

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
print("ESTADÍSTICOS DESCRIPTIVOS - PRIMERA ETAPA FF3 ANUAL")
print("=" * 70)

resumen = pd.DataFrame({
    "Media":    [res["beta_MKT"].mean(), res["beta_SMB"].mean(), res["beta_HML"].mean()],
    "Mediana":  [res["beta_MKT"].median(), res["beta_SMB"].median(), res["beta_HML"].median()],
    "Mínimo":   [res["beta_MKT"].min(), res["beta_SMB"].min(), res["beta_HML"].min()],
    "Máximo":   [res["beta_MKT"].max(), res["beta_SMB"].max(), res["beta_HML"].max()],
    "Desv tip": [res["beta_MKT"].std(), res["beta_SMB"].std(), res["beta_HML"].std()],
}, index=["beta_MKT", "beta_SMB", "beta_HML"])
print(resumen.round(3))

print(f"\nR² medio (FF3):       {res['r2'].mean():.4f}")
print(f"R² medio ajustado:    {res['r2_adj'].mean():.4f}")

# ========================================================================
# ALFAS DE JENSEN
# ========================================================================
print("\n" + "=" * 70)
print("ALFAS DE JENSEN BAJO FF3 ANUAL")
print("=" * 70)
print(f"Alpha medio anual (%):      {res['alpha'].mean() * 100:.3f}")
print(f"|Alpha| medio anual (%):    {res['alpha'].abs().mean() * 100:.3f}")
print(f"Alfas significativos al 5%: {(res['alpha_t'].abs() > 1.96).sum()} / 100")
print(f"Alfas significativos al 1%: {(res['alpha_t'].abs() > 2.58).sum()} / 100")

# ========================================================================
# COMPARACIÓN CAPM vs FF3 (ANUAL)
# ========================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN CAPM vs FF3 - PRIMERA ETAPA (DATOS ANUALES)")
print("=" * 70)

capm_anual = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_anual.pkl"))

comparacion = pd.DataFrame({
    "Métrica": [
        "R² medio",
        "R² mediano",
        "|alpha| medio anual (%)",
        "Alfas significativos al 5% (de 100)",
        "Alfas significativos al 1% (de 100)",
    ],
    "CAPM anual": [
        f"{capm_anual['r2'].mean():.4f}",
        f"{capm_anual['r2'].median():.4f}",
        f"{capm_anual['alpha'].abs().mean() * 100:.3f}",
        f"{(capm_anual['alpha_t'].abs() > 1.96).sum()}",
        f"{(capm_anual['alpha_t'].abs() > 2.58).sum()}",
    ],
    "FF3 anual": [
        f"{res['r2'].mean():.4f}",
        f"{res['r2'].median():.4f}",
        f"{res['alpha'].abs().mean() * 100:.3f}",
        f"{(res['alpha_t'].abs() > 1.96).sum()}",
        f"{(res['alpha_t'].abs() > 2.58).sum()}",
    ],
})
print(comparacion.to_string(index=False))

# ========================================================================
# COMPARACIÓN FF3 MENSUAL vs FF3 ANUAL
# ========================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN FF3: MENSUAL vs ANUAL (PRIMERA ETAPA)")
print("=" * 70)

ff3_mens = pd.read_pickle(os.path.join("datos_procesados", "primera_etapa_ff3.pkl"))

comp_freq = pd.DataFrame({
    "Métrica": [
        "R² medio",
        "|alpha| medio anual (%)",
        "Alfas significativos al 5% (de 100)",
        "beta_MKT media",
        "beta_SMB media",
        "beta_HML media",
    ],
    "FF3 mensual": [
        f"{ff3_mens['r2'].mean():.4f}",
        f"{ff3_mens['alpha'].abs().mean() * 12 * 100:.3f}",
        f"{(ff3_mens['alpha_t'].abs() > 1.96).sum()}",
        f"{ff3_mens['beta_MKT'].mean():.3f}",
        f"{ff3_mens['beta_SMB'].mean():.3f}",
        f"{ff3_mens['beta_HML'].mean():.3f}",
    ],
    "FF3 anual": [
        f"{res['r2'].mean():.4f}",
        f"{res['alpha'].abs().mean() * 100:.3f}",
        f"{(res['alpha_t'].abs() > 1.96).sum()}",
        f"{res['beta_MKT'].mean():.3f}",
        f"{res['beta_SMB'].mean():.3f}",
        f"{res['beta_HML'].mean():.3f}",
    ],
})
print(comp_freq.to_string(index=False))

# ========================================================================
# TOP 5 ALFAS EXTREMOS
# ========================================================================
print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MAYOR ALFA - FF3 ANUAL")
print("=" * 70)
top5 = res.nlargest(5, "alpha")[["alpha", "alpha_t", "beta_MKT", "beta_SMB", "beta_HML", "r2"]].copy()
top5["alpha"] = top5["alpha"] * 100
print(top5.round(3))

print("\n" + "=" * 70)
print("TOP 5 CARTERAS CON MENOR ALFA - FF3 ANUAL")
print("=" * 70)
bot5 = res.nsmallest(5, "alpha")[["alpha", "alpha_t", "beta_MKT", "beta_SMB", "beta_HML", "r2"]].copy()
bot5["alpha"] = bot5["alpha"] * 100
print(bot5.round(3))

# ========================================================================
# GUARDADO
# ========================================================================
res.to_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_ff3_anual.pkl"))
res.to_csv(os.path.join(CARPETA_RESULTADOS, "primera_etapa_ff3_anual.csv"))

print(f"\n✓ Resultados guardados en '{CARPETA_ANUAL}/primera_etapa_ff3_anual.pkl'")
print("✓ Listos para ejecutar paso2_ff3_anual_seccion_cruzada.py")
