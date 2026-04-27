# Data (protected; not included)

This repository does **not** include raw audio, participant-level clinical labels, or private annotations.

To rerun the full pipeline locally, you need access to protected study data and must place it outside the repo (recommended) or under ignored paths:

- `data/raw/` (ignored): raw smartphone PCG WAV files
- `data/interim/` (ignored): intermediate manifests / joins
- `data/processed/` (ignored): model-ready derived datasets

The code expects you to provide explicit paths via CLI arguments or environment variables.

If you are an assessor/reviewer without access to protected data, you can still:
- inspect the code in `src/`
- view the final dissertation figures in `figures/`
- view aggregate, anonymized tables in `results/`
