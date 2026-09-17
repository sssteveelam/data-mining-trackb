export const MAX_WAFER_DIMENSION = 512;

const allowedValues = new Set([0, 1, 2]);

function unwrapPayload(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (payload && typeof payload === "object" && Array.isArray(payload.wafer_map)) {
    return payload.wafer_map;
  }

  return null;
}

/**
 * Validate a JSON payload and return a defensive copy of its wafer map.
 *
 * The API accepts categorical matrices only. Keeping this validation in the
 * browser lets users fix a file before making a network request.
 */
export function validateWaferMap(payload, maxDimension = MAX_WAFER_DIMENSION) {
  const matrix = unwrapPayload(payload);

  if (!matrix || matrix.length === 0) {
    return {
      valid: false,
      error: "JSON phải chứa ma trận wafer_map không rỗng.",
    };
  }

  if (!Array.isArray(matrix[0]) || matrix[0].length === 0) {
    return {
      valid: false,
      error: "Ma trận wafer_map phải có ít nhất một cột.",
    };
  }

  const rows = matrix.length;
  const columns = matrix[0].length;

  if (rows > maxDimension || columns > maxDimension) {
    return {
      valid: false,
      error: `Kích thước tối đa là ${maxDimension}×${maxDimension}. File hiện tại là ${rows}×${columns}.`,
    };
  }

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    const row = matrix[rowIndex];

    if (!Array.isArray(row) || row.length !== columns) {
      return {
        valid: false,
        error: `Ma trận không hợp lệ: hàng ${rowIndex + 1} không cùng số cột.`,
      };
    }

    for (let columnIndex = 0; columnIndex < columns; columnIndex += 1) {
      const value = row[columnIndex];
      if (!Number.isInteger(value) || !allowedValues.has(value)) {
        return {
          valid: false,
          error: `Giá trị tại hàng ${rowIndex + 1}, cột ${columnIndex + 1} phải là số nguyên 0, 1 hoặc 2.`,
        };
      }
    }
  }

  return {
    valid: true,
    matrix: matrix.map((row) => row.slice()),
    rows,
    columns,
  };
}

export function parseWaferMapJson(text) {
  let payload;

  try {
    payload = JSON.parse(text);
  } catch {
    return {
      valid: false,
      error: "Không thể đọc JSON. Hãy kiểm tra dấu phẩy, ngoặc vuông và mã hóa file.",
    };
  }

  return validateWaferMap(payload);
}

export function getMatrixStats(matrix) {
  if (!matrix?.length || !matrix[0]?.length) {
    return { rows: 0, columns: 0, defectRatio: 0 };
  }

  let defects = 0;
  let validPixels = 0;
  const rows = matrix.length;
  const columns = matrix[0].length;

  matrix.forEach((row) => {
    row.forEach((value) => {
      if (value > 0) validPixels += 1;
      if (value === 2) defects += 1;
    });
  });

  return {
    rows,
    columns,
    // Match backend.preprocessing.calculate_defect_ratio: background (0)
    // cells are excluded from the denominator.
    defectRatio: validPixels ? defects / validPixels : 0,
  };
}
