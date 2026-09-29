// Small dense matrix helpers for the TrueSkill match quality formula.
// Matrices are arrays of rows. Sizes here are (#players x #teams), so tiny.

export const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));

export function multiply(A, B) {
  const n = A.length;
  const m = B[0].length;
  const k = B.length;
  const out = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) out[i][j] += a * B[p][j];
    }
  return out;
}

export const add = (A, B) => A.map((row, i) => row.map((x, j) => x + B[i][j]));
export const scale = (A, s) => A.map((row) => row.map((x) => x * s));

/** Determinant via LU decomposition with partial pivoting. */
export function determinant(M) {
  const A = M.map((r) => r.slice());
  const n = A.length;
  let det = 1;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (A[p][c] === 0) return 0;
    if (p !== c) {
      [A[p], A[c]] = [A[c], A[p]];
      det = -det;
    }
    det *= A[c][c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return det;
}

/** Inverse via Gauss Jordan elimination. Throws on a singular matrix. */
export function inverse(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-15) throw new RangeError('matrix is singular');
    [A[p], A[c]] = [A[c], A[p]];
    const piv = A[c][c];
    for (let k = 0; k < 2 * n; k++) A[c][k] /= piv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      if (f === 0) continue;
      for (let k = 0; k < 2 * n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((row) => row.slice(n));
}
