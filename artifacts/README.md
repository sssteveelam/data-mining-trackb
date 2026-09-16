# Model artifacts

The runtime expects a trained bundle in this directory:

```text
model.keras
labels.json
preprocess.json
metrics.json
manifest.json
```

Generate it offline in Colab/Kaggle:

```bash
python -m training.export_model \
  --dataset /path/to/LSWMD.pkl \
  --output-dir artifacts \
  --model-version cnn-9class-v1
```

The repository intentionally does not include the 2+ GB source pickle or model
weights. Until a bundle is generated, the API reports `model_available=false`
and `/predict` returns HTTP 503 rather than fabricating predictions.

