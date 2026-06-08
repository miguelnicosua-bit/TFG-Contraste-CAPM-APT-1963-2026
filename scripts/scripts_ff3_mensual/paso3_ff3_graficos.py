"""
PASO 3 FF3 - Gráficos del modelo de tres factores
TFG Valoración de Activos - Modelo de tres factores Fama-French (1993)
Autor: Miguel Suárez Crespo

Genera cinco figuras en alta resolución (300 dpi) para insertar en el TFG:
  1. Heatmap 10x10 de alfas bajo FF3 (debe ser mucho más blanco que en el CAPM).
  2. Comparación lado a lado: alfas CAPM vs alfas FF3.
  3. Heatmap 10x10 de las betas a SMB (factor tamaño).
  4. Heatmap 10x10 de las betas a HML (factor valor).
  5. Comparación de t-estadísticos de los alfas: CAPM vs FF3.
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

# ========================================================================
# CARGA DE DATOS
# ========================================================================
CARPETA_PROCESADOS = "datos_procesados"
CARPETA_GRAFICOS = "resultados/graficos/ff3_mensual"
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

primera_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))
primera_capm = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 100,
})

# ========================================================================
# FUNCIÓN AUXILIAR: PARSEAR NOMBRES DE CARTERAS A COORDENADAS (size, bm)
# ========================================================================
def parsear_nombre(nombre):
    n = nombre.strip()
    if n == "SMALL LoBM": return 1, 1
    if n == "SMALL HiBM": return 1, 10
    if n == "BIG LoBM":   return 10, 1
    if n == "BIG HiBM":   return 10, 10
    m = re.match(r"ME(\d+)\s+BM(\d+)", n)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Nombre no reconocido: {nombre}")

# Construimos DataFrames con coordenadas size, bm
parsed = pd.DataFrame(
    [parsear_nombre(c) for c in primera_ff3.index],
    columns=["size", "bm"], index=primera_ff3.index
)
ff3 = pd.concat([primera_ff3, parsed], axis=1)
capm = pd.concat([primera_capm, parsed], axis=1)

# ========================================================================
# FUNCIÓN AUXILIAR: CONSTRUIR MATRIZ 10x10 A PARTIR DE UNA SERIE
# ========================================================================
def matriz_10x10(df, columna):
    M = np.full((10, 10), np.nan)
    for _, fila in df.iterrows():
        i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
        M[i, j] = fila[columna]
    return M

# ========================================================================
# GRÁFICO 1: HEATMAP DE ALFAS BAJO FF3
# ========================================================================
matriz_alfa_ff3 = matriz_10x10(ff3, "alpha") * 12 * 100  # anualizado

fig, ax = plt.subplots(figsize=(9, 7))
# Misma escala que el heatmap de alfas del CAPM (para comparación visual directa)
vmax_capm = np.nanmax(np.abs(matriz_10x10(capm, "alpha") * 12 * 100))
norm = TwoSlopeNorm(vmin=-vmax_capm, vcenter=0, vmax=vmax_capm)
im = ax.imshow(matriz_alfa_ff3, cmap="RdYlGn", norm=norm, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_alfa_ff3[i, j]
        color_texto = "white" if abs(valor) > vmax_capm * 0.6 else "black"
        ax.text(j, i, f"{valor:.1f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = small   ←→   10 = big)")
ax.set_title("Alfas anualizados (%) bajo el modelo de tres factores\n"
             "(Escala fija para comparar con el heatmap del CAPM)")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Alfa FF3 anualizado (%)", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "06_heatmap_alfas_ff3_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 1 guardado: 06_heatmap_alfas_ff3_mensual.svg")

# ========================================================================
# GRÁFICO 2: COMPARACIÓN LADO A LADO ALFAS CAPM vs FF3
# ========================================================================
matriz_alfa_capm = matriz_10x10(capm, "alpha") * 12 * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

vmax_global = max(np.nanmax(np.abs(matriz_alfa_capm)),
                  np.nanmax(np.abs(matriz_alfa_ff3)))
norm_g = TwoSlopeNorm(vmin=-vmax_global, vcenter=0, vmax=vmax_global)

# Panel A: CAPM
im1 = ax1.imshow(matriz_alfa_capm, cmap="RdYlGn", norm=norm_g, aspect="equal")
for i in range(10):
    for j in range(10):
        v = matriz_alfa_capm[i, j]
        c = "white" if abs(v) > vmax_global * 0.6 else "black"
        ax1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=c)
ax1.set_xticks(range(10)); ax1.set_yticks(range(10))
ax1.set_xticklabels([f"BM{j+1}" for j in range(10)], fontsize=9)
ax1.set_yticklabels([f"ME{i+1}" for i in range(10)], fontsize=9)
ax1.set_xlabel("Decil BM (growth ←→ value)")
ax1.set_ylabel("Decil Tamaño (pequeña ←→ grande)")
ax1.set_title("(a) Alfas bajo el CAPM")

# Panel B: FF3
im2 = ax2.imshow(matriz_alfa_ff3, cmap="RdYlGn", norm=norm_g, aspect="equal")
for i in range(10):
    for j in range(10):
        v = matriz_alfa_ff3[i, j]
        c = "white" if abs(v) > vmax_global * 0.6 else "black"
        ax2.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=c)
ax2.set_xticks(range(10)); ax2.set_yticks(range(10))
ax2.set_xticklabels([f"BM{j+1}" for j in range(10)], fontsize=9)
ax2.set_yticklabels([f"ME{i+1}" for i in range(10)], fontsize=9)
ax2.set_xlabel("Decil BM (growth ←→ value)")
ax2.set_title("(b) Alfas bajo el modelo de tres factores")

# Una barra de color común
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label("Alfa anualizado (%)", rotation=270, labelpad=20)

fig.suptitle("")
plt.savefig(os.path.join(CARPETA_GRAFICOS, "07_comparacion_alfas_ff3_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 2 guardado: 07_comparacion_alfas_ff3_mensual.svg")

# ========================================================================
# GRÁFICO 3: HEATMAP DE LAS BETAS A SMB
# ========================================================================
matriz_smb = matriz_10x10(ff3, "beta_SMB")

fig, ax = plt.subplots(figsize=(9, 7))
vmax_smb = max(abs(np.nanmax(matriz_smb)), abs(np.nanmin(matriz_smb)))
norm_smb = TwoSlopeNorm(vmin=-vmax_smb, vcenter=0, vmax=vmax_smb)
im = ax.imshow(matriz_smb, cmap="RdBu_r", norm=norm_smb, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_smb[i, j]
        color_texto = "white" if abs(valor) > vmax_smb * 0.6 else "black"
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = pequeña   ←→   10 = grande)")
ax.set_title("")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("$\\hat{\\beta}_{SMB}$", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "08_heatmap_betas_SMB_ff3_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 3 guardado: 08_heatmap_betas_SMB_ff3_mensual.svg")

# ========================================================================
# GRÁFICO 4: HEATMAP DE LAS BETAS A HML
# ========================================================================
matriz_hml = matriz_10x10(ff3, "beta_HML")

fig, ax = plt.subplots(figsize=(9, 7))
vmax_hml = max(abs(np.nanmax(matriz_hml)), abs(np.nanmin(matriz_hml)))
norm_hml = TwoSlopeNorm(vmin=-vmax_hml, vcenter=0, vmax=vmax_hml)
im = ax.imshow(matriz_hml, cmap="RdBu_r", norm=norm_hml, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_hml[i, j]
        color_texto = "white" if abs(valor) > vmax_hml * 0.6 else "black"
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = small   ←→   10 = big)")
ax.set_title("")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("$\\hat{\\beta}_{HML}$", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "09_heatmap_betas_HML_ff3_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 4 guardado: 09_heatmap_betas_HML_ff3_mensual.svg")

# ========================================================================
# GRÁFICO 5: COMPARACIÓN DE T-ESTADÍSTICOS DE ALFAS CAPM vs FF3
# ========================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

t_capm = capm["alpha_t"]
t_ff3 = ff3["alpha_t"]

ax.hist([t_capm, t_ff3], bins=20, label=["CAPM", "FF3"],
        color=["lightcoral", "steelblue"], edgecolor="black", alpha=0.75)

ax.axvline(-1.96, color="red", linestyle="--", linewidth=1.5,
           label="Umbrales $\\pm 1.96$")
ax.axvline(1.96, color="red", linestyle="--", linewidth=1.5)
ax.axvline(0, color="black", linewidth=1)

n_capm = (t_capm.abs() > 1.96).sum()
n_ff3 = (t_ff3.abs() > 1.96).sum()
texto = (f"Alfas significativos al 5%:\n"
         f"  CAPM: {n_capm}/100\n"
         f"  FF3:  {n_ff3}/100")
ax.text(0.02, 0.97, texto, transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="gray", alpha=0.9))

ax.set_xlabel("t-estadístico de $\\hat{\\alpha}_j$")
ax.set_ylabel("Número de carteras")
ax.set_title("Distribución de los t-estadísticos de los alfas: CAPM vs FF3")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "10_comparacion_tstats_ff3_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 5 guardado: 10_comparacion_tstats_ff3_mensual.svg")

print(f"\n✓ Los 5 gráficos del FF3 están en '{CARPETA_GRAFICOS}/'")
print("✓ Resolución 300 dpi, listos para insertar en el TFG")
