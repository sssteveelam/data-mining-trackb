import { describe, expect, it } from "vitest";
import {
  getMatrixStats,
  parseWaferMapJson,
  validateWaferMap,
} from "./validation.js";

describe("validateWaferMap", () => {
  it("accepts a direct matrix and returns a defensive copy", () => {
    const source = [[0, 1], [1, 2]];
    const result = validateWaferMap(source);

    expect(result.valid).toBe(true);
    expect(result.rows).toBe(2);
    expect(result.columns).toBe(2);
    expect(result.matrix).toEqual(source);
    expect(result.matrix).not.toBe(source);
  });

  it("accepts the documented wafer_map object shape", () => {
    const result = parseWaferMapJson('{"wafer_map":[[0,1],[1,2]]}');
    expect(result.valid).toBe(true);
    expect(result.matrix).toEqual([[0, 1], [1, 2]]);
  });

  it("rejects ragged rows and values outside 0, 1, 2", () => {
    expect(validateWaferMap([[0, 1], [1]])).toMatchObject({
      valid: false,
    });
    expect(validateWaferMap([[0, 3]])).toMatchObject({
      valid: false,
    });
  });

  it("rejects an empty or oversized matrix", () => {
    expect(validateWaferMap([])).toMatchObject({ valid: false });
    expect(validateWaferMap([[0]], 0)).toMatchObject({ valid: false });
  });

  it("calculates the defect ratio", () => {
    expect(getMatrixStats([[0, 1], [2, 2]])).toMatchObject({
      rows: 2,
      columns: 2,
      defectRatio: 0.5,
    });
  });
});
