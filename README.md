# geostats-colab-lab

Repositorio “Colab-ready” para exploración geoestadística (EDA), variografía y análisis de anisotropía con datos de perforaciones. Incluye notebooks livianos y módulos reutilizables para cálculo y visualización, con un flujo pensado para Google Colab.

## Propósito
- Estandarizar un flujo reproducible de EDA geoestadístico.
- Proveer funciones mínimas pero correctas para variografía omnidireccional y direccional.
- Facilitar la generación posterior de módulos reutilizables (por ejemplo, por Codex).

## Estructura del repositorio
```
.
├─ data_sample/
│  └─ sample_drillholes.csv
├─ notebooks/
│  ├─ 01_eda_geo.ipynb
│  ├─ 02_variography.ipynb
│  ├─ 03_anisotropy_analysis.ipynb
│  └─ 04_block_model_support.ipynb
├─ src/
│  ├─ anisotropy.py
│  ├─ eda_geo.py
│  ├─ utils_spatial.py
│  └─ variography.py
├─ requirements.txt
└─ README.md
```

## Etapas del flujo (diagrama/tabla)
| Etapa | Notebook / CLI | Qué se hace | Outputs esperados |
| --- | --- | --- | --- |
| 1. EDA | `notebooks/01_eda_geo.ipynb` | Revisión estadística básica, histogramas, QQ-plot y dispersión XY. | Figuras de distribución y tablas de resumen. |
| 2. Variografía | `notebooks/02_variography.ipynb` | Cálculo de variograma experimental y ajuste de modelos. | Curvas de variograma y parámetros ajustados. |
| 3. Anisotropía | `notebooks/03_anisotropy_analysis.ipynb` | Análisis direccional y comparación de estructuras. | Mapas/plots direccionales y conclusiones de anisotropía. |
| 4. Soporte / Block model | `notebooks/04_block_model_support.ipynb` | Ajuste de soporte para block model (upscaling). | Resúmenes de soporte y check de coherencia. |
| 5. Pipeline end-to-end | `python -m src.pipeline config/project.json` | EDA + variografía + kriging 2D + validación simple. | `outputs/figures/`, `outputs/tables/`, `outputs/models/`, `outputs/logs/`. |

## Quickstart en Google Colab
1) Clonar el repo en `/content`:
```bash
git clone <repo_url> /content/geostats-colab-lab
```
2) Instalar dependencias:
```bash
pip install -r /content/geostats-colab-lab/requirements.txt
```
3) Agregar `src` al `sys.path`:
```python
import sys
sys.path.append('/content/geostats-colab-lab/src')
```
4) Importar funciones de ejemplo:
```python
from eda_geo import basic_stats
from variography import experimental_variogram
```

## Cómo correr notebooks o CLI
### Notebooks
1) Instalar dependencias:
```bash
pip install -r requirements.txt
```
2) Levantar Jupyter y abrir los notebooks:
```bash
jupyter lab
```
3) Ejecutar en orden sugerido: `01_eda_geo.ipynb` → `02_variography.ipynb` → `03_anisotropy_analysis.ipynb` → `04_block_model_support.ipynb`.

### CLI (pipeline)
1) Configurar el archivo `config/project.json` con rutas y parámetros.
2) Ejecutar el pipeline:
```bash
python -m src.pipeline config/project.json
```
3) Los resultados se escriben en `outputs/` (figuras, tablas, modelos y logs).

## Datos esperados
Los notebooks y funciones asumen un `DataFrame` con columnas típicas:
- `X`, `Y`, `Z`: coordenadas en metros.
- `grade`: variable de ley (numérica).
- `domain`: dominio o categoría geológica.
- `lithology`: litología o descripción textual.

## Convenciones de código
- Separar cálculos y plotting en funciones distintas.
- Evitar rutas absolutas; usar rutas relativas al repo.
- Documentar unidades (metros, grados, etc.).
- Preferir funciones puras y testeables.
- Mantener comentarios breves en español.

## Outputs versionados
- Se versiona la carpeta `outputs/eda/` con un `.gitkeep` para preservar la estructura del repo.
- Los outputs generados por notebooks o pipeline se guardan en `outputs/` pero no se versionan por defecto (para evitar binarios pesados y resultados específicos de cada corrida).
- Si necesitás versionar un output (por ejemplo, un modelo final o una tabla de referencia), movelo a una carpeta explícita de artefactos y documentalo en el README.

## Advertencia sobre kriging y recuperables
El kriging entrega una **estimación suavizada** (minimiza varianza), por lo que **subestima la variabilidad** real. Para recuperar distribuciones y métricas de **recuperables**, se requiere **simulación geoestadística** (p. ej. SGS) o técnicas equivalentes; no alcanza con el mapa krigeado.

## When generating code with Codex
- Separar **cálculo** de **visualización** (plotting) en funciones distintas.
- Devolver outputs numéricos además de gráficos (p. ej., `lags`, `gamma`, `npairs`).
- Documentar unidades y supuestos en docstrings.
- Priorizar funciones puras y testeables que no dependan del estado global.
- Mantener dependencias razonables y compatibles con Colab.
