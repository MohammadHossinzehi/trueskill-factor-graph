// Gaussian helpers used by every rating system in this package.
//
// TrueSkill does all of its bookkeeping in the *natural* (precision) form of a
// Gaussian: pi = 1 / sigma^2 and tau = mu * pi. In that form multiplying and
// dividing densities is just adding and subtracting the parameters, which is
// exactly what message passing on a factor graph needs.

const SQRT2 = Math.SQRT2;
const SQRT2PI = Math.sqrt(2 * Math.PI);

/**
 * Complementary error function (Numerical Recipes "erfcc").
 * Fractional error below 1.2e-7 everywhere, and crucially it stays *relatively*
 * accurate deep in the tail, which the truncation factor depends on.
 */
export function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + z / 2);
  const r =
    t *
    Math.exp(
      -z * z - 1.26551223 +
        t * (1.00002368 +
        t * (0.37409196 +
        t * (0.09678418 +
        t * (-0.18628806 +
        t * (0.27886807 +
        t * (-1.13520398 +
        t * (1.48851587 +
        t * (-0.82215223 +
        t * 0.17087277))))))))
    );
  return x >= 0 ? r : 2 - r;
}

/** Inverse of erfc, refined with two Halley steps. */
export function erfcinv(y) {
  if (y >= 2) return -100;
  if (y <= 0) return 100;
  const lower = y < 1;
  if (!lower) y = 2 - y;
  const t = Math.sqrt(-2 * Math.log(y / 2));
  let x = -0.70711 * ((2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t);
  for (let i = 0; i < 2; i++) {
    const err = erfc(x) - y;
    x += err / (1.1283791670955126 * Math.exp(-x * x) - x * err);
  }
  return lower ? x : -x;
}

export const pdf = (x, mu = 0, sigma = 1) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (SQRT2PI * sigma);

export const cdf = (x, mu = 0, sigma = 1) => 0.5 * erfc(-(x - mu) / (sigma * SQRT2));

export const ppf = (p, mu = 0, sigma = 1) => mu - sigma * SQRT2 * erfcinv(2 * p);

/** An immutable Gaussian stored in natural parameters. pi = 0 is the uniform "no information" message. */
export class Gaussian {
  constructor(pi = 0, tau = 0) {
    this.pi = pi;
    this.tau = tau;
  }

  static fromMuSigma(mu, sigma) {
    if (!(sigma > 0)) throw new RangeError(`sigma must be positive, got ${sigma}`);
    const pi = 1 / (sigma * sigma);
    return new Gaussian(pi, pi * mu);
  }

  get mu() {
    return this.pi === 0 ? 0 : this.tau / this.pi;
  }

  get sigma() {
    return this.pi === 0 ? Infinity : Math.sqrt(1 / this.pi);
  }

  mul(other) {
    return new Gaussian(this.pi + other.pi, this.tau + other.tau);
  }

  div(other) {
    return new Gaussian(this.pi - other.pi, this.tau - other.tau);
  }

  toString() {
    return `N(mu=${this.mu.toFixed(3)}, sigma=${this.sigma.toFixed(3)})`;
  }
}

/**
 * Additive and multiplicative corrections for a Gaussian truncated by a win
 * (x > margin) or a draw (|x| <= margin). See Herbrich, Minka and Graepel (2007).
 */
export function vWin(t, e) {
  const x = t - e;
  const denom = cdf(x);
  return denom > 1e-300 ? pdf(x) / denom : -x;
}

export function wWin(t, e) {
  const x = t - e;
  const v = vWin(t, e);
  const w = v * (v + x);
  // Mathematically 0 < w < 1; clamp so floating point never makes 1 - w <= 0.
  return Math.min(Math.max(w, 1e-12), 1 - 1e-12);
}

export function vDraw(t, e) {
  const abs = Math.abs(t);
  const a = e - abs;
  const b = -e - abs;
  const denom = cdf(a) - cdf(b);
  const numer = pdf(b) - pdf(a);
  return (denom > 1e-300 ? numer / denom : a) * (t < 0 ? -1 : 1);
}

export function wDraw(t, e) {
  const abs = Math.abs(t);
  const a = e - abs;
  const b = -e - abs;
  const denom = cdf(a) - cdf(b);
  if (!(denom > 0)) throw new RangeError('draw margin too small for this difference');
  const v = vDraw(abs, e);
  const w = v * v + (a * pdf(a) - b * pdf(b)) / denom;
  return Math.min(Math.max(w, 1e-12), 1 - 1e-12);
}
