# Geostats Workspace (Python + GeostatsPy)

Este workspace contiene un pipeline reproducible para pasar de CSV de sondajes a EDA ? variograf?a ? grilla ? kriging ? validaci?n ? export.

## 1) Crear y activar entorno
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install geostatspy numpy pandas matplotlib scipy scikit-learn jupyter
pip install ipywidgets
```

## 2) Datos
- Copi? tus CSV a `csv/`.
- Edit? `config/project.json` para mapear columnas.

## 3) Configuraci?n principal
Archivo: `config/project.json`
- `data_csv_path`: ruta relativa al CSV.
- `columns`: mapea columnas (x, y, z, variable_objetivo, domain).
- `nodata_values`: valores a tratar como nulos.
- `topcut`: activable si necesit?s capar outliers.
- `grid`: permite auto desde extents (usa `dx,dy,dz`) o manual.
- `variogram`: par?metros b?sicos del experimental.
- `kriging`: radio de b?squeda y min/max samples.

## 4) Notebooks (orden recomendado)
1. `notebooks/00_setup_check.ipynb`
2. `notebooks/01_eda.ipynb`
3. `notebooks/02_variography.ipynb`
4. `notebooks/03_grid_model.ipynb`
5. `notebooks/04_kriging.ipynb`
6. `notebooks/05_validation.ipynb`

## 4.1) VS Code: seleccionar kernel correcto (.venv)
Pasos:
1. Abrí un notebook.
2. Click en el selector de kernel (arriba a la derecha).
3. Elegí `geostats (.venv)` (o el Python dentro de `.venv\Scripts\python.exe`).

Si el kernelspec se pierde:
```powershell
.\.venv\Scripts\python -m ipykernel install --user --name geostats-venv --display-name "geostats (.venv)"
```

## 5) Pipeline completo
```powershell
.\.venv\Scripts\activate
python -m src.pipeline config/project.json
```
Si el pipeline falla con `ModuleNotFoundError: geostatspy`, casi seguro no activaste `.venv`.

Opcional (Windows):
```powershell
.\tools\run_pipeline.ps1
```

## 6) Outputs
Todo se guarda en `outputs/`:
- `outputs/figures/`
- `outputs/tables/`
- `outputs/models/`
- `outputs/logs/`

## 7) Demo data
Si no existe el CSV indicado, se genera `csv/demo_points.csv` autom?ticamente con `src/make_demo_data.py`.

---

Notas:
- No se usan ejecutables GSLIB.
- GeostatsPy requiere `numba` y `tqdm` para algunas funciones; este proyecto incluye un shim interno para evitar instalar dependencias extra.
