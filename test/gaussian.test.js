import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Gaussian, cdf, ppf, pdf, erfc, vWin, wWin, vDraw, wDraw } from '../src/gaussian.js';

const close = (a, b, tol = 1e-6, msg) => assert.ok(Math.abs(a - b) <= tol, msg ?? `${a} != ${b}`);

test('cdf matches known standard normal values', () => {
  close(cdf(0), 0.5);
  close(cdf(1), 0.8413447, 1e-6);
  close(cdf(-1.96), 0.0249979, 1e-6);
  close(cdf(3), 0.9986501, 1e-6);
});

test('ppf inverts cdf across the range', () => {
  for (const p of [1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-6]) close(cdf(ppf(p)), p, 1e-8);
  close(ppf(0.975), 1.959964, 1e-5);
});

test('erfc stays relatively accurate in the far tail', () => {
  // erfc(6) = 2.1519736712498913e-17
  close(erfc(6) / 2.1519736712498913e-17, 1, 1e-6);
});

test('pdf integrates to one (trapezoid rule)', () => {
  let s = 0;
  for (let x = -10; x < 10; x += 0.001) s += (pdf(x) + pdf(x + 0.001)) * 0.0005;
  close(s, 1, 1e-9);
});

test('Gaussian multiply and divide are inverse operations', () => {
  const a = Gaussian.fromMuSigma(3, 2);
  const b = Gaussian.fromMuSigma(-1, 0.5);
  const back = a.mul(b).div(b);
  close(back.mu, 3);
  close(back.sigma, 2);
  // product of densities: precision adds
  close(a.mul(b).pi, 1 / 4 + 1 / 0.25);
  assert.equal(new Gaussian().mu, 0);
  assert.equal(new Gaussian().sigma, Infinity);
  assert.throws(() => Gaussian.fromMuSigma(0, 0), RangeError);
});

test('truncation corrections match brute force moments of a truncated normal', () => {
  // For X ~ N(t, 1) truncated to X > e: E[X] = t + v, Var[X] = 1 - w.
  const moments = (t, lo, hi) => {
    let z = 0, m1 = 0, m2 = 0;
    for (let x = t - 12; x < t + 12; x += 1e-4) {
      if (x < lo || x > hi) continue;
      const d = pdf(x, t, 1);
      z += d; m1 += d * x; m2 += d * x * x;
    }
    const mean = m1 / z;
    return { mean, variance: m2 / z - mean * mean };
  };
  for (const [t, e] of [[0.3, 0.2], [-1.5, 0.1], [2, 0]]) {
    const m = moments(t, e, Infinity);
    close(t + vWin(t, e), m.mean, 1e-3);
    close(1 - wWin(t, e), m.variance, 1e-3);
  }
  for (const [t, e] of [[0.3, 0.5], [-0.8, 0.4]]) {
    const m = moments(t, -e, e);
    close(t + vDraw(t, e), m.mean, 1e-3);
    close(1 - wDraw(t, e), m.variance, 1e-3);
  }
});

test('win correction does not blow up for huge upsets', () => {
  const v = vWin(-40, 0.5);
  const w = wWin(-40, 0.5);
  assert.ok(Number.isFinite(v) && v > 0);
  assert.ok(w > 0 && w < 1);
});
