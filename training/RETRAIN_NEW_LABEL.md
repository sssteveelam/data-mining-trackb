# Retrain model with a reviewed new label

The current Kaggle notebook has only two automatic `Horizontal_Stripes`
candidate rows. Do not train a ten-class model from those two rows.

## 1. Review and collect candidate maps

After the notebook has produced `cure_clusters`, `outlier_global_idx`, and
`df_final_large`, inspect the maps in each candidate cluster. Select only maps
that a domain reviewer confirms belong to the same new pattern:

```python
# Replace this with the cluster IDs accepted by the reviewer.
accepted_cluster_ids = [3, 4]

reviewed_global_indices = []
for cluster_id in accepted_cluster_ids:
    reviewed_global_indices.extend(
        outlier_global_idx[index]
        for index in cure_clusters[cluster_id]
    )

df_new_label_reviewed = df_final_large.iloc[reviewed_global_indices].copy()
df_new_label_reviewed["failureType"] = "Horizontal_Stripes"

print(
    "Reviewed Horizontal_Stripes rows:",
    len(df_new_label_reviewed),
)
```

Do not accept a whole cluster blindly. Remove maps that do not show the same
pattern and add independently reviewed examples until the class has enough
samples. The exporter requires at least 30 rows per class and the report
should preferably contain substantially more.

## 2. Build and save the ten-class table

```python
cols = ["waferMap_resized", "failureType"]
df_train_v2_reviewed = pd.concat(
    [
        df_subset[cols],
        df_new_label_reviewed[cols],
    ],
    ignore_index=True,
)

df_train_v2_reviewed.to_pickle(
    "/kaggle/working/WM811K_train_v2_reviewed.pkl"
)
```

## 3. Audit before training

```python
!python -m training.audit_dataset \
  --dataset /kaggle/working/WM811K_train_v2_reviewed.pkl \
  --output /kaggle/working/label_audit.json
```

Continue only when the output contains:

```json
"ready_for_export": true
```

## 4. Export the ten-class model

```python
!python -m training.export_model \
  --dataset /kaggle/working/WM811K_train_v2_reviewed.pkl \
  --output-dir /kaggle/working/artifacts-10class \
  --model-version cnn-10class-v1
```

The output must contain:

```text
model.keras
labels.json
preprocess.json
metrics.json
manifest.json
```

Verify that `labels.json` contains `Horizontal_Stripes` and that
`metrics.json` contains its precision, recall, and F1-score. Only then replace
the baseline artifact directory used by the backend.

