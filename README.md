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

## When generating code with Codex
- Separar **cálculo** de **visualización** (plotting) en funciones distintas.
- Devolver outputs numéricos además de gráficos (p. ej., `lags`, `gamma`, `npairs`).
- Documentar unidades y supuestos en docstrings.
- Priorizar funciones puras y testeables que no dependan del estado global.
- Mantener dependencias razonables y compatibles con Colab.
