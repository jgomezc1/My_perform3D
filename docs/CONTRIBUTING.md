# Contributing & Session Bootstrap

## Dev Environment
- Python 3.11+, OpenSeesPy installed.
- `pip install -r requirements.txt` (if present)
- `OUT_DIR` defaults to `out/` (see `config.py`).

## Typical Workflow
1) **Phase-1 → Phase-2 → Explicit**
   ```bash
   python phase1_run.py
   python run_pipeline.py
   python generate_explicit_model.py --out out --explicit out/explicit_model.py

## Verification Ritual (post-generation)

After running the normal pipeline, always verify that artifacts and explicit models are internally consistent.

```bash
# Rebuild artifacts and explicit model
python run_pipeline.py
python generate_explicit_model.py --out out --explicit out/explicit_model.py

# Hard check: no element references non-existent nodes
python verify_model.py --check endpoints_exist

# Optional quick probe analysis
python explicit_runtime_check.py
python explicit_static_probe.py --P 1.0 --dir X
