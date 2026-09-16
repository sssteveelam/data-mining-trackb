export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "/api/v1"
).replace(/\/+$/, "");

function getErrorMessage(payload, fallback) {
  if (typeof payload?.detail === "string") {
    return payload.detail;
  }

  if (payload?.detail && typeof payload.detail === "object") {
    if (typeof payload.detail.message === "string") {
      return payload.detail.message;
    }
    if (typeof payload.detail.reason === "string") {
      return payload.detail.reason;
    }
  }

  if (Array.isArray(payload?.detail)) {
    return payload.detail
      .map((item) => item?.msg || item)
      .filter(Boolean)
      .join("; ");
  }

  if (typeof payload?.message === "string") {
    return payload.message;
  }

  return fallback;
}

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });
  } catch {
    throw new Error(
      "Không thể kết nối tới backend. Hãy kiểm tra backend đang chạy và URL API.",
    );
  }

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : null;

  if (!response.ok) {
    throw new Error(
      getErrorMessage(payload, `Backend trả về lỗi HTTP ${response.status}.`),
    );
  }

  return payload;
}

export function getHealth() {
  return request("/health");
}

export function getModelInfo() {
  return request("/model-info");
}

export function predictWaferMap(waferMap, metadata = {}) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({
      wafer_map: waferMap,
      metadata,
    }),
  });
}
