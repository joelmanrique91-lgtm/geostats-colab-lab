# Colab (capa temporal y aislada)

Uso rápido:
1. Abrí `colab/00_bootstrap.ipynb` en Google Colab.
2. Configurá `REPO_URL` (si no tenés el repo ya clonado en `/content/geostats-colab-lab`).
3. Ejecutá **Run all**.

Qué hace:
- monta Drive opcionalmente,
- clona/actualiza repo en `/content/geostats-colab-lab`,
- instala dependencias mínimas desde `colab/requirements_colab.txt`,
- fuerza import desde `src/geostats_pipeline` para evitar conflicto con `geostats_pipeline/` raíz,
- materializa `colab/config.colab.runtime.yml` con rutas absolutas compatibles con Colab,
- corre smoke test real: `run_pipeline(..., stage='setup')`.

Luego podés ejecutar manualmente otras etapas (`qaqc`, `variography`, `kriging`, `validation`, etc.) desde el mismo notebook.

Limitaciones:
- Esta capa no porta una app desktop completa.
- Los entrypoints legacy (`src/pipeline.py`, `geostats_pipeline/run.py`) no son parte del flujo Colab recomendado.
- Si querés persistencia entre sesiones, usá Drive para datos/outputs.
