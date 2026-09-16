# Offline training/export

`data-mining-source.ipynb` is kept as the research record. The runtime does
not execute it. Use `export_model.py` in Colab/Kaggle after making
`LSWMD.pkl` available locally:

```bash
python -m training.export_model \
  --dataset /kaggle/working/LSWMD.pkl \
  --output-dir artifacts
```

The exporter uses a stratified train/validation/test split, preserves the
notebook CNN architecture, and writes the model plus preprocessing metadata
needed by the API.

`Horizontal_Stripes` is deliberately excluded: the notebook found only two
candidate samples and did not retrain a ten-class model.

