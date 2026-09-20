# Deploy Wafer Map Classifier

This guide covers the supported deployment shape:

```text
Browser → frontend/nginx → backend/FastAPI → artifacts/model.keras
```

Training is not part of deployment. Train/export the model in Kaggle or
Colab, then copy only the generated `artifacts/` bundle to the server.

## 1. Required project files

The server must contain:

```text
project/
├── artifacts/
│   ├── model.keras
│   ├── labels.json
│   ├── preprocess.json
│   ├── metrics.json
│   └── manifest.json
├── backend/
├── frontend/
├── docker-compose.prod.yml
└── docker-compose.yml
```

Do not copy `LSWMD.pkl`, `.venv-linux`, `frontend/node_modules`, or
`frontend/dist` to the server.

Check the model bundle before starting:

```bash
test -f artifacts/model.keras
test -f artifacts/labels.json
test -f artifacts/preprocess.json
test -f artifacts/metrics.json
test -f artifacts/manifest.json
```

## 2. Local deployment on Windows

Install Docker Desktop and make sure it is running. From PowerShell:

```powershell
cd "C:\Users\<WindowsUser>\Desktop\data-mining-trackb"
docker compose version
docker compose up --build -d
```

Open:

```text
Frontend: http://localhost:5173
API docs: http://localhost:8000/docs
Health:    http://localhost:8000/api/v1/health
```

Check the API:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

The expected fields are:

```text
status        : ok
model_status  : ready
model_loaded  : True
model_version : cnn-9class-v1
```

View logs:

```powershell
docker compose logs -f backend
docker compose logs -f frontend
```

Stop the demo:

```powershell
docker compose down
```

## 3. Deploy on a Linux server/VPS

The server needs Docker Engine with the Compose plugin, a domain or public IP,
and ports `80` and `443` allowed by the firewall. Copy the release archive to
the server, then:

```bash
mkdir -p ~/wafer-map-classifier
cd ~/wafer-map-classifier
unzip wafer-map-classifier-release-ready.zip
```

If the archive extracts into a nested directory, enter that directory before
continuing. Confirm:

```bash
ls -lh artifacts
docker compose version
```

Start the production compose file:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

The production frontend listens on port `80`:

```text
http://SERVER_IP/
http://SERVER_IP/api/v1/health
```

The backend is intentionally only exposed inside the Docker network. Nginx in
the frontend container proxies `/api/` to FastAPI.

Check status and logs:

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f backend
```

Stop:

```bash
docker compose -f docker-compose.prod.yml down
```

## 4. Domain and HTTPS

The included production compose file provides HTTP on port `80`; it does not
issue TLS certificates. For a public URL, put a reverse proxy such as Caddy,
Nginx, or a managed load balancer in front of the frontend container:

```text
https://your-domain.example → frontend container port 80
```

Set the backend CORS origin to the real frontend origin in a `.env` file beside
`docker-compose.prod.yml`:

```dotenv
FRONTEND_ORIGINS=https://your-domain.example
```

Then recreate the services:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

When frontend and API are served from the same domain through the included
Nginx proxy, the browser calls `/api/v1/...` and no separate public backend
port is required.

## 5. Updating the model

First export and validate a new artifact bundle offline. Keep a backup of the
current bundle:

```bash
mv artifacts artifacts-backup-$(date +%Y%m%d-%H%M%S)
mv artifacts-new artifacts
```

Then restart only the backend so it loads the new model:

```bash
docker compose -f docker-compose.prod.yml up -d --force-recreate backend
```

Verify both:

```bash
curl http://localhost/api/v1/health
curl http://localhost/api/v1/model-info
```

For a reviewed ten-class model, `model-info` must show
`class_count: 10` and `Horizontal_Stripes` in `labels`. Do not edit
`labels.json` manually without a matching `model.keras`.

## 6. Troubleshooting

### `model_loaded: false`

Check that the five artifact files are directly under `artifacts/`, not under
`artifacts/artifacts/`. Then inspect:

```bash
docker compose -f docker-compose.prod.yml logs backend
```

### `MODEL_UNAVAILABLE`

The API is reachable but TensorFlow/model loading failed. Confirm that the
backend image was rebuilt and that `model.keras` is readable:

```bash
docker compose -f docker-compose.prod.yml up --build -d backend
```

### Frontend cannot call the API

Check that backend health is ready and that the frontend container is running:

```bash
docker compose -f docker-compose.prod.yml ps
curl http://localhost/api/v1/health
```

### Build takes a long time

The backend image installs TensorFlow CPU and can be large. This is expected
on the first build. Later starts reuse the Docker image cache.

## 7. Security boundary for this academic demo

- There is no login, user management, database, or request history.
- Do not expose the backend port directly to the public internet.
- Do not upload `LSWMD.pkl` to the server.
- Add HTTPS, authentication, rate limiting, logging, and model governance
  before using the service for real production decisions.
