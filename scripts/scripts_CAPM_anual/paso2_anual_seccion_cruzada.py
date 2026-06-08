"""
PASO 2 ANUAL - Segunda etapa Fama-MacBeth con datos anuales
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Para cada año t, regresamos los excesos de rendimiento de las 100 carteras
sobre las betas estimadas en el Paso 1:

    r_{j,t} - r_{f,t} = gamma_{0,t} + gamma_{1,t} beta_j + u_{j,t}

Luego promediamos los T pares de coeficientes y aplicamos la inferencia
de Fama-MacBeth.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_ANUAL = "datos_procesados_anual"
CARPETA_RESULTADOS = "resultados/tablas/capm_anual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos      = pd.read_pickle(os.path.join(CARPETA_ANUAL, "excesos_carteras_anual.pkl"))
mkt_rf       = pd.read_pickle(os.path.join(CARPETA_ANUAL, "mkt_rf_anual.pkl"))
primera_a    = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_anual.pkl"))

betas = primera_a["beta"]

print(f"Datos cargados:")
print(f"  - Excesos anuales: {excesos.shape}")
print(f"  - Betas estimadas: {len(betas)}")

# ========================================================================
# REGRESIONES DE SECCIÓN CRUZADA AÑO A AÑO
# ========================================================================
X_design = sm.add_constant(betas.rename("beta"))

gammas = []
fechas_t = []

for anio in excesos.index:
    y = excesos.loc[anio]
    datos = pd.concat([y, X_design], axis=1).dropna()
    if len(datos) < 10:
        continue

    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas.append(modelo.params.values)
    fechas_t.append(anio)

gammas = pd.DataFrame(
    gammas,
    columns=["gamma_0", "gamma_1"],
    index=fechas_t
)

print(f"\nNúmero de regresiones transversales anuales: {len(gammas)}")

# ========================================================================
# AGREGACIÓN TEMPORAL: MEDIA Y T-ESTADÍSTICOS DE FAMA-MACBETH
# ========================================================================
T = len(gammas)

resumen = pd.DataFrame(index=gammas.columns)
resumen["media"]            = gammas.mean().values
resumen["desv_tipica"]      = gammas.std().values
resumen["error_estandar"]   = resumen["desv_tipica"] / np.sqrt(T)
resumen["t_stat"]           = resumen["media"] / resumen["error_estandar"]
resumen["p_valor"]          = pd.Series(
    2 * (1 - stats.t.cdf(resumen["t_stat"].abs(), df=T-1)),
    index=resumen.index
)
resumen["media_pct"]        = resumen["media"] * 100  # ya está en términos anuales

print("\n" + "=" * 70)
print("RESULTADOS DE LA SEGUNDA ETAPA FAMA-MACBETH (DATOS ANUALES)")
print("=" * 70)
print(resumen.round(5))

# ========================================================================
# INTERPRETACIÓN
# ========================================================================
prima_mkt = mkt_rf.mean() * 100
print("\n" + "=" * 70)
print("INTERPRETACIÓN")
print("=" * 70)
print(f"  Prima de mercado realizada (% anual):       {prima_mkt:.3f}")
print(f"  gamma_0 estimada (% anual):                 {resumen.loc['gamma_0', 'media_pct']:.3f}")
print(f"  gamma_1 estimada (% anual):                 {resumen.loc['gamma_1', 'media_pct']:.3f}")
print(f"  Predicción CAPM: gamma_0=0, gamma_1=prima")
print(f"")
print(f"  ¿Se rechaza gamma_0 = 0?")
print(f"     t-stat = {resumen.loc['gamma_0', 't_stat']:.3f}, "
      f"p-valor = {resumen.loc['gamma_0', 'p_valor']:.4f}")
print(f"  ¿Se rechaza gamma_1 = 0?")
print(f"     t-stat = {resumen.loc['gamma_1', 't_stat']:.3f}, "
      f"p-valor = {resumen.loc['gamma_1', 'p_valor']:.4f}")

# ========================================================================
# COMPARACIÓN CON LA SEGUNDA ETAPA MENSUAL
# ========================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN SEGUNDA ETAPA: MENSUAL vs ANUAL")
print("=" * 70)

segunda_m = pd.read_pickle(os.path.join("datos_procesados", "segunda_etapa.pkl"))

comparacion = pd.DataFrame({
    "Métrica": [
        "gamma_0 anual (%)",
        "  t-stat",
        "  p-valor",
        "gamma_1 anual (%)",
        "  t-stat",
        "  p-valor",
        "Prima de mercado realizada (% anual)",
    ],
    "Mensual": [
        f"{segunda_m.loc['gamma_0', 'media_anual_%']:.3f}",
        f"{segunda_m.loc['gamma_0', 't_stat']:.3f}",
        f"{segunda_m.loc['gamma_0', 'p_valor']:.4f}",
        f"{segunda_m.loc['gamma_1', 'media_anual_%']:.3f}",
        f"{segunda_m.loc['gamma_1', 't_stat']:.3f}",
        f"{segunda_m.loc['gamma_1', 'p_valor']:.4f}",
        f"{mkt_rf.mean() * 100:.3f}",  # aproximado, son distintos por composición
    ],
    "Anual": [
        f"{resumen.loc['gamma_0', 'media_pct']:.3f}",
        f"{resumen.loc['gamma_0', 't_stat']:.3f}",
        f"{resumen.loc['gamma_0', 'p_valor']:.4f}",
        f"{resumen.loc['gamma_1', 'media_pct']:.3f}",
        f"{resumen.loc['gamma_1', 't_stat']:.3f}",
        f"{resumen.loc['gamma_1', 'p_valor']:.4f}",
        f"{mkt_rf.mean() * 100:.3f}",
    ],
})
print(comparacion.to_string(index=False))

# ========================================================================
# GUARDADO
# ========================================================================
gammas.to_pickle(os.path.join(CARPETA_ANUAL, "gammas_anuales.pkl"))
resumen.to_pickle(os.path.join(CARPETA_ANUAL, "segunda_etapa_anual.pkl"))
resumen.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_anual.csv"))

print(f"\n✓ Resultados guardados en '{CARPETA_ANUAL}/'")
print("✓ Listos para ejecutar paso3_anual_contrastes.py")
