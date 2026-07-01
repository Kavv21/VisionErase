# SAM2 install notes

SAM2 is a required dependency for `workers/segmentation/tasks.py` but is not
listed in `requirements.txt` — it has no PyPI release, so `pip install -r
requirements.txt` alone will not make it available.

## Local dev

```
pip install -e /path/to/sam2 --no-deps
```

`--no-deps` avoids SAM2's own requirements file pulling in hydra-core /
omegaconf versions that conflict with this project's pins. The packages it
actually needs at runtime (`torch`, `numpy`, `Pillow`, etc.) are already in
`requirements.txt`.

SAM2 must be importable via `PYTHONPATH` / editable install from any working
directory — it resolves its Hydra config search path relative to the
installed `sam2` package, not the caller's cwd.

## Docker

Add the same `pip install -e /path/to/sam2 --no-deps` step to the worker
image build (`Dockerfile.worker`) once the checkpoint and config paths are
finalized, then mount/copy the SAM2 source into the image.

## Checkpoint

`workers/segmentation/tasks.py` resolves the checkpoint from
`settings.model_cache_dir`:

- Docker: `/app/model_weights/sam2_hiera_small.pt`
- Local: `~/visionerase/model_weights/sam2_hiera_small.pt`

The config file used is `configs/sam2.1/sam2.1_hiera_s.yaml`, resolved
relative to the installed `sam2` package (not this repo).
