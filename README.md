# Wafer-map classifier demo

This repository turns `data-mining-source.ipynb` into a small local web
application for academic demonstration.

## Architecture

- `training/`: offline preprocessing, CNN training, and artifact export.
- `backend/`: FastAPI inference service.
- `frontend/`: React/Vite Vietnamese UI.
- `artifacts/`: generated model bundle, intentionally excluding the large
  source dataset and weights from version control.
- `examples/sample_wafer.json`: a 64 × 64 WM-811K-shaped categorical sample
  (values `0`, `1`, and `2`) for a realistic preview.

The production-facing model has nine classes:

```text
Center, Donut, Edge-Loc, Edge-Ring, Loc,
Near-full, Normal, Random, Scratch
```

`Horizontal_Stripes` is not included because the notebook created it from two
candidate samples without retraining the classifier.

## Chạy nhanh

Artifact model đã có sẵn trong thư mục `artifacts/`. Mở hai cửa sổ terminal.

### 1. Backend — Ubuntu/WSL

```bash
cd "/mnt/c/Users/LAM QUOC HUY/Desktop/data-mining-trackb"
source .venv-linux/bin/activate
.venv-linux/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Kiểm tra backend từ PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

Kết quả cần có `model_loaded=True`, `model_status=ready` và
`model_version=cnn-9class-v1`.

### 2. Frontend — PowerShell

```powershell
cd "C:\Users\LAM QUOC HUY\Desktop\data-mining-trackb\frontend"
npm install
npm run dev
```

Mở `http://localhost:5173`, sau đó chọn **Dùng mẫu WM-811K 64 × 64** hoặc
upload `examples/sample_wafer.json`.

### Chạy bằng Docker

```powershell
cd "C:\Users\LAM QUOC HUY\Desktop\data-mining-trackb"
docker compose up --build
```

Mở `http://localhost:5173`. Dừng hệ thống bằng:

```powershell
docker compose down
```

## Generate the model bundle

Run this offline in Colab/Kaggle, where `LSWMD.pkl` is available:

```bash
python -m training.export_model \
  --dataset /path/to/LSWMD.pkl \
  --output-dir artifacts \
  --model-version cnn-9class-v1
```

The API intentionally refuses to fabricate predictions when the bundle is
missing. It reports model availability from `/api/v1/health` and
`/api/v1/model-info`.

## Run locally with Docker

After generating `artifacts/model.keras` and its metadata:

```bash
docker compose up --build
```

- Frontend: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/v1/health`

Upload `examples/sample_wafer.json` from the frontend. The API accepts a JSON
body shaped like:

```json
{
  "wafer_map": [[0, 1, 2], [0, 1, 1]],
  "metadata": {
    "lot_name": "lot1",
    "wafer_index": 1
  }
}
```

The bundled sample uses the same square, circular-wafer layout as the
notebook's resized maps. The small matrix above is only a minimal schema
example; it is not intended to look like a physical wafer map.

The input must be a rectangular matrix containing only integer values `0`, `1`
and `2`, with each side no larger than 512 cells. A low top-1 confidence is
returned as `review_required=true`; it is not presented as a definitive
physical-quality conclusion.

## Development checks

Backend:

```bash
pytest backend/tests training/tests
```

Frontend:

```bash
cd frontend
npm install
npm run build
```
