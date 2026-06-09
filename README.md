# Contraste empírico del CAPM y del modelo de tres factores (1963–2026)

> **Trabajo Fin de Grado en Economía** — Universidad de Santiago de Compostela
> **Autor:** Miguel Nicolás Suárez Crespo
> **Director:** Antonio Rodríguez Sampayo
> **Curso académico:** 2025–2026

---

## Resumen

Este repositorio contiene el código y los resultados del contraste empírico del **CAPM** (Sharpe, 1964; Lintner, 1965) y del **modelo de tres factores de Fama y French** (FF3, 1993), llevado a cabo como parte del Trabajo Fin de Grado en Economía.

El análisis replica y extiende la metodología en dos etapas de **Fama y MacBeth (1973)** sobre las 100 carteras *Size × Book-to-Market* construidas por Kenneth R. French, ampliando el período muestral original de Fama y French (1992) hasta abril de 2026. Se realizan los contrastes con **datos mensuales** (`T = 754` observaciones) —análisis principal en el TFG— y con **datos anuales** (`T = 62` años) como ejercicio complementario de robustez.

Los principales hallazgos confirman el **rechazo empírico del CAPM** (intercepto significativamente distinto de cero, prima de mercado sin precio detectable en la sección cruzada, falta de linealidad y violación de la no remuneración del riesgo idiosincrático) y muestran que el **modelo de tres factores mejora sustancialmente** el ajuste, aunque sin agotar la sección cruzada de rendimientos.

---

## Estructura del repositorio

```
TFG-Contraste-CAPM-APT-1963-2026/
│
├── datos/                          Datos primarios (CSVs originales de Kenneth R. French)
│   ├── datos_100_carteras.csv
│   └── datos_factores.csv
│
├── scripts/                        Código fuente del análisis
│   ├── capm_mensual/               Contraste CAPM con datos mensuales
│   ├── capm_anual/                 Contraste CAPM con datos anuales (robustez)
│   ├── ff3_mensual/                Contraste FF3 con datos mensuales
│   └── ff3_anual/                  Contraste FF3 con datos anuales (robustez)
│
├── resultados/                     Tablas y figuras generadas por los scripts
│   ├── graficos/                   Figuras en formato vectorial (SVG)
│   │   ├── capm_mensual/
│   │   ├── capm_anual/
│   │   ├── ff3_mensual/
│   │   └── ff3_anual/
│   └── tablas/                     Tablas exportadas a CSV
│       ├── capm_mensual/
│       ├── capm_anual/
│       ├── ff3_mensual/
│       └── ff3_anual/
│
├── README.md                       Este archivo
├── requirements.txt                Dependencias de Python
└── .gitignore                      Archivos a excluir del control de versiones
```

> **Nota.** Las carpetas `datos_procesados/` y `datos_procesados_anual/` no se incluyen en el repositorio porque contienen archivos intermedios (`.pkl`) que se regeneran automáticamente al ejecutar el paso 0 de cada bloque.

---

## Datos

Los datos provienen de la base pública mantenida por **Kenneth R. French** en el Tuck School of Business (Dartmouth College), accesible en:
[https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

Se utilizan dos archivos:

- **Fama/French 3 Factors** — Exceso de rendimiento de la cartera de mercado, tipo libre de riesgo y factores SMB y HML.
- **100 Portfolios Formed on Size and Book-to-Market (10×10)** — 100 carteras *value-weighted* construidas con todas las acciones cotizadas en NYSE, AMEX y NASDAQ.

**Período muestral:** julio 1963 – abril 2026 (`T = 754` observaciones mensuales; `T = 62` años naturales completos para el análisis anual).

---

## Instalación y ejecución

### Requisitos

- Python ≥ 3.10
- Las dependencias están listadas en `requirements.txt`.

### Instalación

```bash
git clone https://github.com/miguelnicosua-bit/TFG-Contraste-CAPM-APT-1963-2026.git
cd TFG-Contraste-CAPM-APT-1963-2026
pip install -r requirements.txt
```

### Orden de ejecución

Los scripts se ejecutan en orden numérico dentro de cada bloque. El **paso 0 de cada bloque genera los archivos intermedios** (`.pkl`) en las carpetas `datos_procesados/` y `datos_procesados_anual/`, que los pasos siguientes consumen. Estas carpetas se crean automáticamente al ejecutar el código.

Los cuatro bloques tienen dependencias entre sí:

- **CAPM mensual:** independiente, se ejecuta primero.
- **FF3 mensual:** reutiliza `datos_procesados/` (requiere haber ejecutado el bloque CAPM mensual).
- **CAPM anual:** reutiliza `datos_procesados/` (requiere haber ejecutado el bloque CAPM mensual).
- **FF3 anual:** requiere los tres bloques anteriores.

**Orden recomendado:** CAPM mensual → FF3 mensual → CAPM anual → FF3 anual.

Desde la raíz del repositorio:

**Bloque 1 — CAPM mensual:**
```bash
python scripts/capm_mensual/paso0_limpieza.py
python scripts/capm_mensual/paso1_betas.py
python scripts/capm_mensual/paso2_seccion_cruzada.py
python scripts/capm_mensual/paso3_contrastes.py
python scripts/capm_mensual/paso4_graficos.py
python scripts/capm_mensual/paso5_comprobaciones.py
```

**Bloque 2 — FF3 mensual:** análogamente, dentro de `scripts/ff3_mensual/`.
**Bloque 3 — CAPM anual:** análogamente, dentro de `scripts/capm_anual/`.
**Bloque 4 — FF3 anual:** análogamente, dentro de `scripts/ff3_anual/`.

---

## Principales resultados

### CAPM mensual

| Métrica | Valor |
|---|---|
| Beta media | 1,096 |
| R² medio | 0,645 |
| Alfas significativos al 5% (de 100) | 34 |
| γ̄₀ anualizada | 13,79%*** |
| γ̄₁ anualizada | −4,22% (n.s.) |
| Prima de mercado realizada | 7,17% anual |
| **Test GRS** | **F = 2,26 (p < 0,001)** |

Los tres contrastes formales (GRS, linealidad de la SML mediante β² y remuneración del riesgo idiosincrático mediante σ(ε)) rechazan el CAPM al 5%, en línea con Fama y French (1992).

### Modelo de tres factores (FF3) mensual

| Métrica | CAPM | FF3 | Mejora |
|---|---|---|---|
| R² medio | 0,645 | 0,800 | +15,5 pp |
| \|α\| anualizado medio | 2,376% | 1,467% | −38% |
| Alfas significativos al 5% | 34 | 21 | −38% |
| **Test GRS** | **2,26** | **2,15** | Sigue rechazando |

| Coeficiente | Estimación anual | Prima realizada |
|---|---|---|
| γ̄_MKT | −6,43%** | 7,17% |
| γ̄_SMB | 1,52% (n.s.) | 1,77% |
| γ̄_HML | 3,82%*** | 3,56% |

El coeficiente del factor HML coincide aproximadamente con la prima realizada y resulta significativo, validando el efecto valor como factor de riesgo con precio. El modelo de tres factores mejora sustancialmente al CAPM, aunque el test GRS sigue rechazando la hipótesis conjunta de alfas nulos.

### Análisis anual (robustez)

Los resultados con `T = 62` observaciones anuales son **cualitativamente idénticos** a los mensuales: rechazo del CAPM por las mismas vías y mejora análoga al pasar al FF3. La menor potencia estadística se refleja en menos alfas individualmente significativos.

---

## Validación del código

Cada bloque incluye un script de comprobaciones que verifica numéricamente la corrección del código:

- `scripts/capm_mensual/paso5_comprobaciones.py`
- `scripts/capm_anual/comprobaciones_capm_anual.py`
- `scripts/ff3_mensual/paso4_ff3_comprobaciones.py`
- `scripts/ff3_anual/comprobaciones_ff3_anual.py`

Cada script ejecuta cinco tests independientes (identidades algebraicas, replicación manual de las regresiones, replicación paso a paso del test GRS, coherencia entre estimadores y consistencia entre frecuencias). Todos los tests devuelven `✓ PASA`.

---

## Referencias principales

- **Black, F., Jensen, M. C. y Scholes, M. (1972).** "The Capital Asset Pricing Model: Some Empirical Tests". En M. C. Jensen (Ed.), *Studies in the Theory of Capital Markets* (pp. 79–121). New York: Praeger.
- **Fama, E. F. y French, K. R. (1992).** "The Cross-Section of Expected Stock Returns". *Journal of Finance*, 47(2), 427–465.
- **Fama, E. F. y French, K. R. (1993).** "Common Risk Factors in the Returns on Stocks and Bonds". *Journal of Financial Economics*, 33(1), 3–56.
- **Fama, E. F. y MacBeth, J. D. (1973).** "Risk, Return, and Equilibrium: Empirical Tests". *Journal of Political Economy*, 81(3), 607–636.
- **Gibbons, M. R., Ross, S. A. y Shanken, J. (1989).** "A Test of the Efficiency of a Given Portfolio". *Econometrica*, 57(5), 1121–1152.
- **Marín, J. M. y Rubio, G. (2011).** *Economía financiera*. Antoni Bosch Editor.

---

## Cómo citar este trabajo

> Suárez Crespo, M. N. (2026). *Contraste empírico del CAPM y del modelo de tres factores de Fama y French (1963–2026)*. Trabajo Fin de Grado, Facultad de Ciencias Económicas y Empresariales, Universidad de Santiago de Compostela.

---

## Asistencia de IA

El desarrollo de este repositorio ha contado con la asistencia del modelo de inteligencia artificial **Claude (Anthropic)** en tareas concretas:

- Generación inicial y depuración de los scripts en Python para el análisis empírico (regresiones, test GRS, generación de figuras).
- Redacción y revisión estilística de este README.
- El código en Python contenido en este Jupyter Notebook.

Todo el código ha sido posteriormente revisado, ejecutado y validado numéricamente por el autor mediante los scripts de comprobaciones (véase la sección "Validación del código"). Las decisiones metodológicas, la interpretación de los resultados, la elección de los contrastes y la estructura general del análisis son responsabilidad exclusiva del autor.

---

## Licencia

Este repositorio se publica con fines exclusivamente académicos. Los datos pertenecen a sus respectivos titulares (Kenneth R. French, CRSP, COMPUSTAT).
