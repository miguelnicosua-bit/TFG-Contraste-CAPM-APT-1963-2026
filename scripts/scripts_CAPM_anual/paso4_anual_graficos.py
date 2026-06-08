"""
PASO 4 ANUAL - Gráficos del contraste empírico del CAPM (datos anuales)
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Cinco figuras en alta resolución (300 dpi):
  1. SML teórica vs empírica con datos anuales.
  2. Mapa de calor 10x10 de las betas anuales.
  3. Mapa de calor 10x10 de los alfas (anuales, en %).
  4. Mapa de calor 10x10 de los rendimientos medios (anuales, en %).
  5. Comparación de t-estadísticos de alfas: mensual vs anual.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Configuración para gráficos vectoriales con texto seleccionable
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.colors import TwoSlopeNorm

CARPETA_ANUAL = "datos_procesados_anual"
CARPETA_GRAFICOS = "resultados/graficos/capm_anual"
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

primera     = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_anual.pkl"))
mkt_rf      = pd.read_pickle(os.path.join(CARPETA_ANUAL, "mkt_rf_anual.pkl"))
primera_m   = pd.read_pickle(os.path.join("datos_procesados", "primera_etapa.pkl"))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 100,
})

# ========================================================================
# FUNCIÓN AUXILIAR
# ========================================================================
def parsear_nombre(nombre):
    n = nombre.strip()
    if n == "SMALL LoBM": return 1, 1
    if n == "SMALL HiBM": return 1, 10
    if n == "BIG LoBM":   return 10, 1
    if n == "BIG HiBM":   return 10, 10
    m = re.match(r"ME(\d+)\s+BM(\d+)", n)
    return int(m.group(1)), int(m.group(2))

parsed = pd.DataFrame(
    [parsear_nombre(c) for c in primera.index],
    columns=["size", "bm"], index=primera.index
)
datos = pd.concat([primera, parsed], axis=1)

# ========================================================================
# GRÁFICO 1: SML EMPÍRICA VS TEÓRICA (datos anuales)
# ========================================================================
fig, ax = plt.subplots(figsize=(9, 6))

betas = datos["beta"]
rend_anual = datos["rend_medio"] * 100
prima_mkt = mkt_rf.mean() * 100

sc = ax.scatter(betas, rend_anual, c=datos["bm"], cmap="RdYlGn",
                s=70, edgecolors="black", linewidths=0.4, alpha=0.85, zorder=3)

x_teo = np.linspace(0.5, 1.7, 100)
y_teo = x_teo * prima_mkt
ax.plot(x_teo, y_teo, "--", color="black", linewidth=2,
        label=f"SML teórica del CAPM (pendiente = {prima_mkt:.2f}%)", zorder=2)

coef = np.polyfit(betas, rend_anual, 1)
y_emp = np.polyval(coef, x_teo)
ax.plot(x_teo, y_emp, "-", color="red", linewidth=2,
        label=f"SML empírica (pendiente = {coef[0]:.2f}%)", zorder=2)

ax.scatter([1.0], [prima_mkt], marker="*", s=300, color="gold",
           edgecolors="black", linewidths=1.0, zorder=4, label="Cartera de mercado")

ax.set_xlabel("Beta estimada ($\\hat{\\beta}_j$)")
ax.set_ylabel("Exceso de rendimiento medio anual (%)")
ax.set_title("Security Market Line con datos anuales: teórica vs empírica\n"
             "100 carteras Size $\\times$ BE/ME, 1964 – 2025")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(sc, ax=ax, ticks=[1, 5, 10])
cbar.set_label("Decil de Book-to-Market\n(1 = growth, 10 = value)",
               rotation=270, labelpad=35)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "01_SML_CAPM_anual.svg"),
             format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 1 guardado: 01_SML_CAPM_anual.svg")

# ========================================================================
# GRÁFICO 2: HEATMAP 10x10 DE LAS BETAS (anuales)
# ========================================================================
matriz_beta = np.full((10, 10), np.nan)
for _, fila in datos.iterrows():
    i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
    matriz_beta[i, j] = fila["beta"]

fig, ax = plt.subplots(figsize=(9, 7))
vmax_b = max(abs(np.nanmax(matriz_beta) - 1), abs(np.nanmin(matriz_beta) - 1))
norm_b = TwoSlopeNorm(vmin=1 - vmax_b, vcenter=1, vmax=1 + vmax_b)
im = ax.imshow(matriz_beta, cmap="RdBu_r", norm=norm_b, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_beta[i, j]
        dist = abs(valor - 1)
        color_texto = "white" if dist > vmax_b * 0.55 else "black"
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = small   ←→   10 = big)")
ax.set_title("Betas estimadas ($\\hat{\\beta}_j$) por cartera — Datos anuales")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Beta estimada", rotation=270, labelpad=20)
cbar.ax.axhline(1, color="black", linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "02_heatmap_betas_CAPM_anual.svg"),
             format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 2 guardado: 02_heatmap_betas_CAPM_anual.svg")

# ========================================================================
# GRÁFICO 3: HEATMAP 10x10 DE LOS ALFAS (anuales)
# ========================================================================
matriz_alfa = np.full((10, 10), np.nan)
for _, fila in datos.iterrows():
    i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
    matriz_alfa[i, j] = fila["alpha"] * 100

fig, ax = plt.subplots(figsize=(9, 7))
vmax = np.nanmax(np.abs(matriz_alfa))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(matriz_alfa, cmap="RdYlGn", norm=norm, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_alfa[i, j]
        color_texto = "white" if abs(valor) > vmax * 0.6 else "black"
        ax.text(j, i, f"{valor:.1f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = small   ←→   10 = big)")
ax.set_title("Alfas de Jensen anuales (%) por cartera — Datos anuales")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Alfa anual (%)", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "03_heatmap_alfas_CAPM_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 3 guardado: 03_heatmap_alfas_CAPM_anual.svg")

# ========================================================================
# GRÁFICO 4: HEATMAP 10x10 DE RENDIMIENTOS MEDIOS
# ========================================================================
matriz_rend = np.full((10, 10), np.nan)
for _, fila in datos.iterrows():
    i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
    matriz_rend[i, j] = fila["rend_medio"] * 100

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(matriz_rend, cmap="YlOrRd", aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_rend[i, j]
        max_val = np.nanmax(matriz_rend)
        color_texto = "white" if valor > max_val * 0.65 else "black"
        ax.text(j, i, f"{valor:.1f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = small   ←→   10 = big)")
ax.set_title("Exceso de rendimiento medio anual (%) por cartera — Datos anuales")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Rendimiento medio en exceso (%)", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "04_heatmap_rendimientos_CAPM_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 4 guardado: 04_heatmap_rendimientos_CAPM_anual.svg")

# ========================================================================
# GRÁFICO 5: COMPARACIÓN DE T-ESTADÍSTICOS DE ALFAS (mensual vs anual)
# ========================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

t_mens = primera_m["alpha_t"]
t_anual = primera["alpha_t"]

ax.hist([t_mens, t_anual], bins=20,
        label=["Mensual (T = 754)", "Anual (T = 62)"],
        color=["lightcoral", "steelblue"], edgecolor="black", alpha=0.75)

ax.axvline(-1.96, color="red", linestyle="--", linewidth=1.5,
           label="Umbrales $\\pm 1.96$")
ax.axvline(1.96, color="red", linestyle="--", linewidth=1.5)
ax.axvline(0, color="black", linewidth=1)

n_mens = (t_mens.abs() > 1.96).sum()
n_anual = (t_anual.abs() > 1.96).sum()
texto = (f"Alfas significativos al 5%:\n"
         f"  Mensual: {n_mens}/100\n"
         f"  Anual:   {n_anual}/100")
ax.text(0.02, 0.97, texto, transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="gray", alpha=0.9))

ax.set_xlabel("t-estadístico de $\\hat{\\alpha}_j$")
ax.set_ylabel("Número de carteras")
ax.set_title("Distribución de los t-estadísticos de los alfas: mensual vs anual")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "05_tstats_comparacion_CAPM_anual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 5 guardado: 05_tstats_comparacion_CAPM_anual.svg")

print(f"\n✓ Los 5 gráficos anuales están en '{CARPETA_GRAFICOS}/'")
print("✓ Resolución 300 dpi, listos para insertar en el GitHub")
