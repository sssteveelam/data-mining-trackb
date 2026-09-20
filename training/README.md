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

## Promoting a discovered label safely

The notebook's CURE step is a discovery step, not a ground-truth annotation
step.  In the current notebook output, `Horizontal_Stripes` contains only two
candidate maps.  Do not export those two rows as a ten-class production model:
there is not enough data for a defensible stratified train/validation/test
evaluation.

After a human reviews and labels additional candidate maps, save the reviewed
table from the notebook.  The table may contain either the original
`waferMap` column or the notebook's `waferMap_resized` column:

```python
# Run after the reviewed labels have been added to df_train_version2.
df_train_version2.to_pickle("/kaggle/working/WM811K_train_v2_reviewed.pkl")
```

Audit the actual counts before training:

```bash
python -m training.audit_dataset \
  --dataset /kaggle/working/WM811K_train_v2_reviewed.pkl \
  --output /kaggle/working/label_audit.json
```

The audit exits with status `2` when any class has fewer than 30 rows. This is
still only a conservative minimum for the configured stratified split; aim for
substantially more independently reviewed `Horizontal_Stripes` maps before
reporting per-class metrics.

Only when the audit reports `ready_for_export: true`, run:

```bash
python -m training.export_model \
  --dataset /kaggle/working/WM811K_train_v2_reviewed.pkl \
  --output-dir /kaggle/working/artifacts-10class \
  --model-version cnn-10class-v1
```

The exporter infers the class count from `failureType`, writes the actual
class counts to `metrics.json`, and preserves the labels in `labels.json` and
`manifest.json`.  It never invents a metric for a class that was not present
in the reviewed input.
