import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE_URL,
  getHealth,
  getModelInfo,
  predictWaferMap,
} from "./lib/api.js";
import {
  getMatrixStats,
  validateWaferMap,
} from "./lib/validation.js";
import sampleWafer from "./sample_wafer.json";

const EMPTY_HEALTH = {
  state: "unknown",
  label: "Chưa kiểm tra kết nối",
};

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${(Math.max(0, Math.min(1, number)) * 100).toFixed(1)}%`;
}

function formatRatio(value) {
  return formatPercent(value);
}

function normalizeTopK(result) {
  if (!Array.isArray(result?.top_k)) return [];

  return result.top_k
    .filter((item) => item && typeof item.label === "string")
    .slice(0, 3)
    .map((item) => ({
      label: item.label,
      probability: Number(item.probability ?? item.confidence ?? 0),
    }));
}

function normalizeModelVersion(modelInfo) {
  if (!modelInfo || typeof modelInfo !== "object") return null;
  return (
    modelInfo.model_version ||
    modelInfo.version ||
    modelInfo.manifest?.model_version ||
    null
  );
}

function MatrixPreview({ matrix, title, subtitle }) {
  const canvasRef = useRef(null);
  const stats = getMatrixStats(matrix);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrix?.length || !matrix[0]?.length) return;

    // WM-811K maps are square categorical rasters. Keep the preview square so
    // the circular wafer geometry is not stretched into a banner.
    const width = 440;
    const height = 440;
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    canvas.style.aspectRatio = `${width} / ${height}`;

    const context = canvas.getContext("2d");
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.imageSmoothingEnabled = false;
    context.fillStyle = "#000000";
    context.fillRect(0, 0, width, height);

    const rows = matrix.length;
    const columns = matrix[0].length;
    const cellSize = Math.min(width / columns, height / rows);
    const mapWidth = cellSize * columns;
    const mapHeight = cellSize * rows;
    const offsetX = (width - mapWidth) / 2;
    const offsetY = (height - mapHeight) / 2;
    const colors = {
      0: "#000000",
      1: "#008080",
      2: "#ffd700",
    };

    matrix.forEach((row, rowIndex) => {
      row.forEach((value, columnIndex) => {
        const x = offsetX + columnIndex * cellSize;
        const y = offsetY + rowIndex * cellSize;
        const nextX = offsetX + (columnIndex + 1) * cellSize;
        const nextY = offsetY + (rowIndex + 1) * cellSize;
        context.fillStyle = colors[value] || colors[0];
        context.fillRect(x, y, nextX - x, nextY - y);
      });
    });

    // Keep the grid useful for tiny hand-authored matrices, but do not draw
    // it over 64×64 maps: the notebook renders those as uninterrupted pixels.
    if (rows <= 32 && columns <= 32 && cellSize >= 8) {
      context.strokeStyle = "rgba(0, 0, 0, 0.22)";
      context.lineWidth = 0.5;
      for (let rowIndex = 0; rowIndex <= rows; rowIndex += 1) {
        const y = offsetY + rowIndex * cellSize;
        context.beginPath();
        context.moveTo(offsetX, y);
        context.lineTo(offsetX + mapWidth, y);
        context.stroke();
      }
      for (let columnIndex = 0; columnIndex <= columns; columnIndex += 1) {
        const x = offsetX + columnIndex * cellSize;
        context.beginPath();
        context.moveTo(x, offsetY);
        context.lineTo(x, offsetY + mapHeight);
        context.stroke();
      }
    }
  }, [matrix]);

  return (
    <section className="matrix-card">
      <div className="matrix-card-heading">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {matrix?.length > 0 && (
          <span className="matrix-size">
            {stats.rows} × {stats.columns}
          </span>
        )}
      </div>
      {matrix?.length > 0 ? (
        <canvas
          ref={canvasRef}
          className="matrix-canvas"
          role="img"
          aria-label={`${title}, ma trận ${stats.rows} hàng và ${stats.columns} cột`}
        />
      ) : (
        <div className="matrix-empty">
          <span className="empty-icon">＋</span>
          <span>Chưa có dữ liệu để hiển thị</span>
        </div>
      )}
    </section>
  );
}

function Legend() {
  return (
    <div className="legend" aria-label="Chú giải wafer map">
      <span className="legend-title">Chú giải</span>
      <span className="legend-item">
        <i className="legend-swatch legend-background" />
        0 · Nền
      </span>
      <span className="legend-item">
        <i className="legend-swatch legend-die" />
        1 · Wafer/die
      </span>
      <span className="legend-item">
        <i className="legend-swatch legend-defect" />
        2 · Defect
      </span>
    </div>
  );
}

function StatusBadge({ health }) {
  return (
    <span className={`status-badge status-${health.state}`}>
      <i className="status-dot" />
      {health.label}
    </span>
  );
}

function StatCard({ label, value, accent = "" }) {
  return (
    <div className={`stat-card ${accent ? `stat-${accent}` : ""}`}>
      <span className="stat-label">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TopKList({ topK }) {
  if (!topK.length) {
    return <p className="muted">Backend chưa trả về phân bố top-k.</p>;
  }

  return (
    <div className="top-k-list">
      {topK.map((item, index) => (
        <div className="top-k-row" key={`${item.label}-${index}`}>
          <div className="top-k-rank">{index + 1}</div>
          <div className="top-k-main">
            <div className="top-k-label">
              <span>{item.label}</span>
              <strong>{formatPercent(item.probability)}</strong>
            </div>
            <div className="probability-track">
              <span style={{ width: `${Math.max(0, Math.min(1, item.probability)) * 100}%` }} />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function App() {
  const fileInputRef = useRef(null);
  const [matrix, setMatrix] = useState(null);
  const [fileName, setFileName] = useState("");
  const [validationError, setValidationError] = useState("");
  const [apiError, setApiError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [lotName, setLotName] = useState("");
  const [waferIndex, setWaferIndex] = useState("");
  const [health, setHealth] = useState(EMPTY_HEALTH);
  const [modelVersion, setModelVersion] = useState(null);

  const inputStats = useMemo(() => getMatrixStats(matrix), [matrix]);
  const topK = useMemo(() => normalizeTopK(result), [result]);

  const loadPayload = useCallback((payload, name = "") => {
    const checked = validateWaferMap(payload);
    setResult(null);
    setApiError("");

    if (!checked.valid) {
      setMatrix(null);
      setFileName(name);
      setValidationError(checked.error);
      return false;
    }

    setMatrix(checked.matrix);
    setFileName(name);
    setValidationError("");
    if (payload && !Array.isArray(payload) && typeof payload === "object") {
      const metadata = payload.metadata;
      if (metadata && typeof metadata === "object") {
        setLotName(
          typeof metadata.lot_name === "string" ? metadata.lot_name : "",
        );
        setWaferIndex(
          Number.isInteger(metadata.wafer_index)
            ? String(metadata.wafer_index)
            : "",
        );
      }
    }
    return true;
  }, []);

  const loadFile = useCallback(
    async (file) => {
      if (!file) return;

      if (file.size > 5 * 1024 * 1024) {
        setValidationError("File quá lớn. Vui lòng chọn JSON nhỏ hơn 5 MB.");
        setMatrix(null);
        setResult(null);
        return;
      }

      try {
        const text = await file.text();
        loadPayload(JSON.parse(text), file.name);
      } catch {
        setMatrix(null);
        setResult(null);
        setFileName(file.name);
        setValidationError(
          "Không thể đọc file JSON. Hãy kiểm tra nội dung và mã hóa UTF-8.",
        );
      }
    },
    [loadPayload],
  );

  useEffect(() => {
    let active = true;

    Promise.allSettled([getHealth(), getModelInfo()]).then(([healthResult, modelResult]) => {
      if (!active) return;

      if (healthResult.status === "fulfilled") {
        setHealth({
          state: "online",
          label:
            healthResult.value?.status === "ok"
              ? "Backend đang hoạt động"
              : "Backend đã kết nối",
        });
      } else {
        setHealth({
          state: "offline",
          label: "Chưa kết nối backend",
        });
      }

      if (modelResult.status === "fulfilled") {
        setModelVersion(normalizeModelVersion(modelResult.value));
      }
    });

    return () => {
      active = false;
    };
  }, []);

  const handleFileChange = (event) => {
    loadFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    loadFile(event.dataTransfer.files?.[0]);
  };

  const loadSample = () => {
    loadPayload(sampleWafer.wafer_map, "sample-wafer-map-64x64.json");
  };

  const handleAnalyze = async () => {
    if (!matrix || isLoading) return;

    setIsLoading(true);
    setApiError("");
    setResult(null);

    try {
      const metadata = {};
      if (lotName.trim()) metadata.lot_name = lotName.trim();
      if (waferIndex.trim()) {
        const parsedIndex = Number(waferIndex);
        metadata.wafer_index = Number.isFinite(parsedIndex)
          ? parsedIndex
          : waferIndex.trim();
      }

      const prediction = await predictWaferMap(matrix, metadata);
      setResult(prediction);
    } catch (error) {
      setApiError(error.message || "Không thể phân tích wafer map.");
    } finally {
      setIsLoading(false);
    }
  };

  const processedMap = result?.processed_map;
  const displayedModelVersion =
    result?.model_version || modelVersion || "Chưa xác định";

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-inner">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              <span />
              <span />
              <span />
              <span />
            </div>
            <div>
              <p className="eyebrow">DATA MINING · TRACK B</p>
              <h1>Wafer Map Classifier</h1>
            </div>
          </div>
          <div className="header-meta">
            <StatusBadge health={health} />
            <span className="model-chip">Model · {displayedModelVersion}</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="intro-section">
          <div>
            <p className="eyebrow accent-eyebrow">PHÂN TÍCH HÌNH ẢNH</p>
            <h2>Nhận diện mẫu hình lỗi trên wafer map</h2>
            <p className="intro-copy">
              Tải lên ma trận JSON gồm các giá trị <code>0</code>,{" "}
              <code>1</code> và <code>2</code> để mô hình CNN dự đoán mẫu hình
              lỗi. Đây là tín hiệu hỗ trợ phân tích, không thay thế đánh giá
              vật lý tại dây chuyền.
            </p>
          </div>
          <div className="intro-note">
            <span className="note-icon">i</span>
            <span>Input được resize nearest-neighbor về 64 × 64 trước khi dự đoán.</span>
          </div>
        </section>

        <div className="workspace-grid">
          <section className="panel input-panel">
            <div className="panel-heading">
              <div>
                <span className="step-number">01</span>
                <div>
                  <h2>Chọn wafer map</h2>
                  <p>
                    Ma trận categorical 0/1/2, tối đa 512 × 512; mẫu demo là
                    map WM-811K 64 × 64.
                  </p>
                </div>
              </div>
              <button className="text-button" type="button" onClick={loadSample}>
                Dùng mẫu WM-811K 64 × 64
              </button>
            </div>

            <input
              ref={fileInputRef}
              className="sr-only"
              type="file"
              accept=".json,application/json"
              onChange={handleFileChange}
            />
            <button
              className={`drop-zone ${isDragging ? "drop-zone-active" : ""}`}
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                setIsDragging(false);
              }}
              onDrop={handleDrop}
            >
              <span className="upload-icon" aria-hidden="true">
                ↑
              </span>
              <span className="drop-title">
                {fileName || "Kéo thả file JSON vào đây"}
              </span>
              <span className="drop-subtitle">
                hoặc nhấn để chọn file từ máy tính
              </span>
            </button>

            {validationError && (
              <div className="message message-error" role="alert">
                <span>!</span>
                {validationError}
              </div>
            )}

            {matrix && (
              <>
                <div className="input-summary">
                  <span>
                    <strong>{inputStats.rows} × {inputStats.columns}</strong> ô dữ liệu
                  </span>
                  <span>
                    Tỷ lệ defect: <strong>{formatRatio(inputStats.defectRatio)}</strong>
                  </span>
                </div>

                <MatrixPreview
                  matrix={matrix}
                  title="Preview đầu vào"
                  subtitle={fileName || "Ma trận wafer map"}
                />
                <Legend />

                <div className="metadata-fields">
                  <label>
                    <span>Lot name <em>tuỳ chọn</em></span>
                    <input
                      type="text"
                      placeholder="Ví dụ: LOT-2026-01"
                      value={lotName}
                      onChange={(event) => setLotName(event.target.value)}
                    />
                  </label>
                  <label>
                    <span>Wafer index <em>tuỳ chọn</em></span>
                    <input
                      type="text"
                      inputMode="numeric"
                      placeholder="Ví dụ: 12"
                      value={waferIndex}
                      onChange={(event) => setWaferIndex(event.target.value)}
                    />
                  </label>
                </div>

                <button
                  className="primary-button analyze-button"
                  type="button"
                  onClick={handleAnalyze}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <span className="spinner" /> Đang phân tích…
                    </>
                  ) : (
                    <>
                      Phân tích wafer map <span aria-hidden="true">→</span>
                    </>
                  )}
                </button>
              </>
            )}

            {!matrix && !validationError && (
              <div className="format-help">
                <span className="format-help-icon">{`{ }`}</span>
                <div>
                  <strong>Định dạng mẫu</strong>
                  <code>
                    {`{
  "wafer_map": [[0, 1, 1], [0, 2, 1]]
}`}
                  </code>
                </div>
              </div>
            )}
          </section>

          <section className="panel result-panel">
            <div className="panel-heading">
              <div>
                <span className="step-number">02</span>
                <div>
                  <h2>Kết quả phân tích</h2>
                  <p>Nhãn dự đoán và độ tin cậy của mô hình.</p>
                </div>
              </div>
              {result && <span className="result-ready">Đã hoàn tất</span>}
            </div>

            {apiError && (
              <div className="message message-error result-error" role="alert">
                <span>!</span>
                {apiError}
              </div>
            )}

            {!result && !apiError && (
              <div className="result-empty">
                <div className="result-empty-art" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                </div>
                <h3>Chưa có kết quả</h3>
                <p>
                  Chọn một wafer map ở bên trái rồi nhấn “Phân tích wafer map”
                  để bắt đầu.
                </p>
              </div>
            )}

            {result && (
              <div className="result-content">
                {result.review_required && (
                  <div className="review-warning" role="status">
                    <span className="warning-icon">!</span>
                    <div>
                      <strong>Độ tin cậy thấp — cần review</strong>
                      <p>
                        Kết quả top-1 chỉ là tín hiệu tham khảo. Hãy kiểm tra
                        lại wafer map và đánh giá bổ sung.
                      </p>
                    </div>
                  </div>
                )}

                <div className="prediction-hero">
                  <div>
                    <span className="result-label">Mẫu hình lỗi dự đoán</span>
                    <h3>{result.predicted_label || "Unknown"}</h3>
                  </div>
                  <div className="confidence-ring">
                    <strong>{formatPercent(result.confidence)}</strong>
                    <span>confidence</span>
                  </div>
                </div>

                <div className="stats-grid">
                  <StatCard
                    label="Tỷ lệ defect"
                    value={formatRatio(result.defect_ratio)}
                    accent="rose"
                  />
                  <StatCard
                    label="Kích thước input"
                    value={
                      result.input_shape?.length === 2
                        ? `${result.input_shape[0]} × ${result.input_shape[1]}`
                        : `${inputStats.rows} × ${inputStats.columns}`
                    }
                  />
                  <StatCard
                    label="Sau tiền xử lý"
                    value={
                      result.processed_shape?.length === 2
                        ? `${result.processed_shape[0]} × ${result.processed_shape[1]}`
                        : "64 × 64"
                    }
                  />
                </div>

                <div className="top-k-section">
                  <div className="section-title-row">
                    <h3>Top-3 dự đoán</h3>
                    <span>Softmax score</span>
                  </div>
                  <TopKList topK={topK} />
                </div>

                {processedMap && (
                  <MatrixPreview
                    matrix={processedMap}
                    title="Ma trận sau tiền xử lý"
                    subtitle="Resize nearest-neighbor · 64 × 64"
                  />
                )}

                <div className="result-footer">
                  <span>
                    Model version: <strong>{displayedModelVersion}</strong>
                  </span>
                  <span className="footer-disclaimer">
                    Confidence là tín hiệu tham khảo, chưa calibration.
                  </span>
                </div>
              </div>
            )}
          </section>
        </div>

        <footer className="app-footer">
          <span>
            Backend API: <code>{API_BASE_URL}</code>
          </span>
          <span>9 lớp phân loại · CPU inference · Không lưu dữ liệu</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
