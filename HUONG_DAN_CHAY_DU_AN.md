# Hướng dẫn chạy dự án Wafer Map Classifier

> Cập nhật: 13/09/2026  
> Mục tiêu: chạy demo backend FastAPI và frontend React/Vite cho mô hình phân loại wafer map.

## 1. Tổng quan

Dự án gồm:

- `training/`: script train và export model offline.
- `artifacts/`: model và metadata dùng cho inference.
- `backend/`: API FastAPI.
- `frontend/`: giao diện React/Vite tiếng Việt.
- `examples/sample_wafer.json`: mẫu 64 × 64 theo cấu trúc wafer map WM-811K
  (giá trị phân loại `0`, `1`, `2`) để preview giống dữ liệu notebook.
- `data-mining-source.ipynb`: notebook nghiên cứu ban đầu.

Artifact hiện tại là model baseline 9 lớp:

```text
Center, Donut, Edge-Loc, Edge-Ring, Loc,
Near-full, Normal, Random, Scratch
```

`Horizontal_Stripes` không được đưa vào model vì notebook mới phát hiện 2 mẫu và chưa retrain classifier.

Code training đã có sẵn đường đi để export model 10 lớp sau khi có dữ liệu
`Horizontal_Stripes` được chuyên gia duyệt. Không được chỉ sửa tay
`labels.json`: phải retrain CNN, đánh giá lớp mới và export lại toàn bộ artifact.

## 2. Yêu cầu

### Bắt buộc

- Windows 10/11.
- WSL2 với Ubuntu.
- Python 3.13 trong Ubuntu.
- Node.js và npm trên Windows.
- Artifact model đã export từ Kaggle.

### Kiểm tra công cụ

Trong PowerShell:

```powershell
node --version
npm --version
wsl -l -v
```

Trong Ubuntu:

```bash
python3 --version
```

Không dùng Python 3.14 để cài TensorFlow. Nếu Ubuntu mặc định là Python 3.14, dùng `uv` để tạo môi trường Python 3.13 như phần 4.

## 3. Chuẩn bị model artifact từ Kaggle

Backend không tự train model và không cần file dataset 2 GB khi chạy. Cần có 5 file sau:

```text
artifacts/
├── model.keras
├── labels.json
├── preprocess.json
├── metrics.json
└── manifest.json
```

Các file này được tạo trong Kaggle tại:

```text
/kaggle/working/artifacts/
```

Tải `artifacts.zip` về máy, giải nén vào thư mục project:

```text
C:\Users\<Tên người dùng>\Desktop\data-mining-trackb\artifacts\
```

Kiểm tra trên Windows:

```powershell
Get-ChildItem "C:\Users\<Tên người dùng>\Desktop\data-mining-trackb\artifacts"
```

`model.keras` phải nằm trực tiếp trong `artifacts`, không được lồng thành:

```text
artifacts\artifacts\model.keras
```

### Nếu cần hoàn thành phần nhãn mới

Trong Kaggle, sau khi rà soát và bổ sung đủ mẫu `Horizontal_Stripes`, lưu
dataset đã duyệt:

```python
df_train_version2.to_pickle(
    "/kaggle/working/WM811K_train_v2_reviewed.pkl"
)
```

Kiểm tra số lượng từng nhãn:

```bash
python -m training.audit_dataset \
  --dataset /kaggle/working/WM811K_train_v2_reviewed.pkl \
  --output /kaggle/working/label_audit.json
```

Chỉ train khi audit báo `ready_for_export: true`:

```bash
python -m training.export_model \
  --dataset /kaggle/working/WM811K_train_v2_reviewed.pkl \
  --output-dir /kaggle/working/artifacts-10class \
  --model-version cnn-10class-v1
```

Artifact 10 lớp phải có `Horizontal_Stripes` trong `labels.json`, có metrics
riêng cho lớp mới trong `metrics.json`, rồi mới được chép vào thư mục runtime.

## 4. Cài backend bằng WSL2

Mở Ubuntu và đi tới project Windows:

```bash
cd "/mnt/c/Users/<Tên người dùng>/Desktop/data-mining-trackb"
```

### 4.1. Cài Python và công cụ hệ thống

```bash
sudo apt update
sudo apt install -y curl ca-certificates python3-venv python3-pip
```

### 4.2. Tạo Python 3.13 bằng uv

Nếu `python3 --version` là 3.14, cài `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Tải Python 3.13:

```bash
uv python install 3.13
```

Tạo virtual environment:

```bash
rm -rf .venv-linux
uv venv --python 3.13 --seed .venv-linux
source .venv-linux/bin/activate
```

Kiểm tra:

```bash
python --version
```

Kết quả cần là `Python 3.13.x`.

### 4.3. Cài dependency backend

```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

Kiểm tra TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Cảnh báo CUDA nếu máy không có GPU là bình thường. Backend vẫn chạy bằng CPU.

## 5. Chạy backend

Trong Ubuntu, với virtual environment đã active:

```bash
python -m uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000
```

Giữ cửa sổ Ubuntu này mở.

### 5.1. Kiểm tra health

Mở PowerShell Windows mới:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

Kết quả cần có:

```text
status        : ok
model_status  : ready
model_loaded  : True
model_version : cnn-9class-v1
```

### 5.2. Mở tài liệu API

```text
http://localhost:8000/docs
```

API chính:

```text
GET  /api/v1/health
GET  /api/v1/model-info
POST /api/v1/predict
```

### 5.3. Test dự đoán bằng PowerShell

```powershell
$body = @{
    wafer_map = @(
        @(0, 1, 2),
        @(0, 1, 1)
    )
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/predict `
  -ContentType "application/json" `
  -Body $body
```

Response hợp lệ sẽ có:

```text
predicted_label
confidence
review_required
top_k
processed_map
defect_ratio
```

## 6. Chạy frontend

Mở PowerShell khác:

```powershell
cd "C:\Users\<Tên người dùng>\Desktop\data-mining-trackb\frontend"
npm install
npm run dev
```

Mở trình duyệt:

```text
http://localhost:5173
```

Upload file mẫu:

```text
C:\Users\<Tên người dùng>\Desktop\data-mining-trackb\examples\sample_wafer.json
```

File mẫu là ma trận vuông 64 × 64, giữ hình học đĩa wafer và bảng mã màu
`0` (nền), `1` (vùng wafer/die), `2` (defect) như các map đã resize trong
notebook. Nút **Dùng dữ liệu mẫu** trên giao diện tải cùng một mẫu này.

Frontend hỗ trợ:

- Upload hoặc kéo-thả JSON.
- Kiểm tra ma trận chữ nhật.
- Chỉ chấp nhận giá trị `0`, `1`, `2`.
- Hiển thị preview ma trận.
- Hiển thị nhãn dự đoán, top-3, confidence và defect ratio.
- Cảnh báo `cần review` nếu confidence thấp.

## 7. Định dạng input JSON

Dạng tối thiểu:

```json
{
  "wafer_map": [
    [0, 1, 2],
    [0, 1, 1]
  ]
}
```

Có thể gửi thêm metadata:

```json
{
  "wafer_map": [
    [0, 1, 2],
    [0, 1, 1]
  ],
  "metadata": {
    "lot_name": "lot1",
    "wafer_index": 1
  }
}
```

Quy tắc:

- Ma trận phải có ít nhất một hàng và một cột.
- Tất cả hàng phải có cùng số cột.
- Giá trị chỉ được là số nguyên `0`, `1`, `2`.
- Mỗi chiều tối đa `512`.
- Backend resize ma trận về `64×64` bằng nearest-neighbor.

Mẫu `examples/sample_wafer.json` đã ở kích thước 64 × 64 nên preview không bị
biến thành một ma trận đồ chơi nhỏ; các ma trận khác vẫn được backend resize
theo cùng quy tắc.

## 8. Chạy frontend/backend bằng Docker

Docker Desktop phải được cài và đang chạy.

Từ thư mục project trên PowerShell:

```powershell
cd "C:\Users\<Tên người dùng>\Desktop\data-mining-trackb"
docker compose up --build
```

Mở:

```text
Frontend: http://localhost:5173
API docs: http://localhost:8000/docs
Health:    http://localhost:8000/api/v1/health
```

Dừng hệ thống:

```powershell
docker compose down
```

## 9. Lỗi thường gặp

### `No matching distribution found for tensorflow`

Bạn đang dùng Python 3.14 hoặc Python không có wheel TensorFlow phù hợp. Kiểm tra:

```bash
python --version
```

Phải dùng Python 3.13 trong `.venv-linux`.

### `No module named pip`

Nếu environment được tạo bằng `uv` không có pip, dùng:

```bash
uv pip install --python .venv-linux/bin/python \
  -r backend/requirements.txt
```

Hoặc tạo lại environment có pip:

```bash
rm -rf .venv-linux
uv venv --python 3.13 --seed .venv-linux
source .venv-linux/bin/activate
```

### `model_loaded: False`

Kiểm tra:

```bash
ls -lh artifacts/model.keras
```

Nếu file không tồn tại, đặt lại đầy đủ 5 artifact vào thư mục `artifacts`.

### `MODEL_UNAVAILABLE`

Backend đã chạy nhưng không load được model. Xem chi tiết:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

Đọc trường `detail` để biết thiếu file, thiếu TensorFlow hay artifact không tương thích.

### HTTP 422 khi dự đoán

JSON không đúng schema. Kiểm tra:

- Có khóa `wafer_map`.
- Ma trận không rỗng.
- Các hàng cùng kích thước.
- Không có giá trị ngoài `0`, `1`, `2`.

### Frontend không kết nối backend

Đảm bảo backend vẫn đang chạy ở port `8000`. Khi chạy frontend bằng `npm run dev`, Vite proxy mặc định chuyển `/api` tới:

```text
http://localhost:8000
```

## 10. Chạy test

Backend và training:

```bash
python -m pytest -q backend/tests training/tests
```

Frontend:

```powershell
cd frontend
npm run test
npm run build
```

## 11. Giới hạn của bản demo

- Chỉ nhận wafer map dạng ma trận JSON, không nhận ảnh PNG/JPG tự do.
- Không có đăng nhập, database, lịch sử hoặc batch processing.
- Confidence là tín hiệu tham khảo, chưa calibration.
- Kết quả là dự đoán mẫu hình wafer map, không phải kết luận chất lượng vật lý tuyệt đối.
- Training phải chạy offline trên Kaggle/Colab; backend chỉ load artifact và inference.
