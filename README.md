# geostats-colab-lab

Repositorio “Colab-ready” para exploración geoestadística (EDA), variografía, declustering, block model y kriging con datos de perforaciones. El flujo se orquesta desde notebooks livianos y un paquete Python (`src/geostats_pipeline`) que centraliza la lógica reproducible.

## Propósito
- Estandarizar un flujo reproducible de EDA geoestadístico con QA/QC, variografía y kriging.
- Centralizar la lógica en `src/geostats_pipeline` y dejar los notebooks como interfaz.
- Garantizar trazabilidad por corrida con `outputs/run_YYYYMMDD_HHMM/manifest.json`.

## Estructura del repositorio
```
.
├─ configs/
│  └─ config.yml
├─ data_sample/
│  └─ sample_drillholes.csv
├─ notebooks/
│  ├─ 00_setup_check.ipynb
│  ├─ 01_data_qaqc.ipynb
│  ├─ 02_compositing_declustering.ipynb
│  ├─ 03_eda_domain_spatial.ipynb
│  ├─ 04_variography.ipynb
│  ├─ 05_block_model.ipynb
│  ├─ 06_estimation_ok.ipynb
│  ├─ 07_validation.ipynb
│  ├─ 08_uncertainty_simulation.ipynb
│  └─ 09_reporting_export.ipynb
├─ src/
│  └─ geostats_pipeline/
│     ├─ io.py
│     ├─ config.py
│     ├─ qaqc.py
│     ├─ support.py
│     ├─ compositing.py
│     ├─ declustering.py
│     ├─ eda.py
│     ├─ variography.py
│     ├─ anisotropy.py
│     ├─ block_model.py
│     ├─ kriging.py
│     ├─ validation.py
│     ├─ simulation.py
│     ├─ reporting.py
│     └─ steps.py
├─ tests/
└─ README.md
```

## Etapas del flujo
| Etapa | Notebook / CLI | Qué se hace | Outputs esperados |
| --- | --- | --- | --- |
| 1. Setup | `notebooks/00_setup_check.ipynb` | Carga + validación de columnas + trazabilidad. | Tablas de resumen. |
| 2. QA/QC | `notebooks/01_data_qaqc.ipynb` | Duplicados, missing, outliers. | Tablas + histogramas. |
| 3. Compositing/Declustering | `notebooks/02_compositing_declustering.ipynb` | Compositado (si hay intervalos) y declustering. | Tablas y stats. |
| 4. EDA espacial | `notebooks/03_eda_domain_spatial.ipynb` | Mapas y estadísticas por dominio. | Figuras. |
| 5. Variografía | `notebooks/04_variography.ipynb` | Variograma experimental y ajuste. | Figura + modelo. |
| 6. Block model | `notebooks/05_block_model.ipynb` | Generación de grilla. | Tabla de grid. |
| 7. Kriging | `notebooks/06_estimation_ok.ipynb` | OK puntual/bloques con varianza. | Tabla + figura. |
| 8. Validación | `notebooks/07_validation.ipynb` | CV + métricas + swaths. | Tablas + figura. |
| 9. Simulación | `notebooks/08_uncertainty_simulation.ipynb` | Simulación (opcional). | Tabla de percentiles. |
| 10. Reporte | `notebooks/09_reporting_export.ipynb` | Manifest y exportes. | `manifest.json`. |

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
4) Importar el pipeline:
```python
from geostats_pipeline.steps import run_pipeline
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
3) Ejecutar en orden sugerido: `00_setup_check.ipynb` → `01_data_qaqc.ipynb` → ... → `09_reporting_export.ipynb`.

### CLI (pipeline)
1) Configurar el archivo `configs/config.yml` con rutas y parámetros.
2) Ejecutar el pipeline por etapas:
```bash
python -m geostats_pipeline.run --config configs/config.yml --stage all
```
3) Los resultados se escriben en `outputs/run_YYYYMMDD_HHMM/` con `manifest.json`.

## Datos esperados
El config exige columnas declaradas en `data`:
- `x_col`, `y_col`, `z_col`: coordenadas.
- `value_col`: variable objetivo.
- `domain_col` opcional para domaining.
- `from_col/to_col` si se trabaja con intervalos.

Si el config referencia columnas inexistentes, el loader falla con un error claro que lista las columnas disponibles.

## Outputs
- Los outputs generados por notebooks o pipeline se guardan en `outputs/run_*/`.
- No se versionan por defecto (para evitar binarios pesados y resultados específicos de cada corrida).

## Advertencia sobre kriging y recuperables
El kriging entrega una **estimación suavizada** (minimiza varianza), por lo que **subestima la variabilidad** real. Para recuperar distribuciones y métricas de **recuperables**, se requiere **simulación geoestadística** (p. ej. SGS) o técnicas equivalentes.

## Nota sobre dataset_original.csv
El config por defecto referencia `dataset_original.csv` y la variable `Magsus`. Si el archivo no está presente en el entorno, el loader fallará con un error claro. Ajustá `data.path` y `value_col` según tu dataset real.
