// A tiny factor graph with Gaussian messages: just enough machinery for
// TrueSkill's expectation propagation schedule.
import { Gaussian } from './gaussian.js';

/** Change between two Gaussians, used as the convergence test. */
function delta(a, b) {
  const piDelta = Math.abs(a.pi - b.pi);
  if (piDelta === Infinity) return 0;
  return Math.max(Math.abs(a.tau - b.tau), Math.sqrt(piDelta));
}

/**
 * A variable node. Its marginal (`value`) is always the product of every
 * message it has received, so we keep the messages around per factor and
 * swap one out at a time.
 */
export class Variable {
  constructor(name = '') {
    this.name = name;
    this.value = new Gaussian();
    this.messages = new Map();
  }

  attach(factor) {
    this.messages.set(factor, new Gaussian());
  }

  message(factor) {
    return this.messages.get(factor);
  }

  _set(value) {
    const d = delta(this.value, value);
    this.value = value;
    return d;
  }

  /** Replace the message coming from `factor` and refresh the marginal. */
  updateMessage(factor, msg) {
    const old = this.messages.get(factor);
    this.messages.set(factor, msg);
    return this._set(this.value.div(old).mul(msg));
  }

  /** Force the marginal to `value`, back solving the message that explains it. */
  updateValue(factor, value) {
    const old = this.messages.get(factor);
    this.messages.set(factor, value.mul(old).div(this.value));
    return this._set(value);
  }
}

class Factor {
  constructor(vars) {
    this.vars = vars;
    for (const v of vars) v.attach(this);
  }
}

/** Prior N(mu, sigma^2 + tau^2): the rating before the match plus skill drift. */
export class PriorFactor extends Factor {
  constructor(variable, prior, dynamic) {
    super([variable]);
    this.prior = prior;
    this.dynamic = dynamic;
  }

  down() {
    const sigma = Math.sqrt(this.prior.sigma ** 2 + this.dynamic ** 2);
    return this.vars[0].updateValue(this, Gaussian.fromMuSigma(this.prior.mu, sigma));
  }
}

/** performance ~ N(skill, beta^2): connects a skill to a noisy per match performance. */
export class LikelihoodFactor extends Factor {
  constructor(mean, value, variance) {
    super([mean, value]);
    this.mean = mean;
    this.valueVar = value;
    this.variance = variance;
  }

  _send(from, to) {
    const msg = from.value.div(from.message(this));
    const a = 1 / (1 + this.variance * msg.pi);
    return to.updateMessage(this, new Gaussian(a * msg.pi, a * msg.tau));
  }

  down() {
    return this._send(this.mean, this.valueVar);
  }

  up() {
    return this._send(this.valueVar, this.mean);
  }
}

/** sum = sum_i coeffs[i] * terms[i]. Used for team performance and pairwise team differences. */
export class SumFactor extends Factor {
  constructor(sum, terms, coeffs) {
    super([sum, ...terms]);
    this.sum = sum;
    this.terms = terms;
    this.coeffs = coeffs;
  }

  _update(target, vars, coeffs) {
    let piInv = 0;
    let mu = 0;
    for (let i = 0; i < vars.length; i++) {
      const div = vars[i].value.div(vars[i].message(this));
      mu += coeffs[i] * div.mu;
      if (piInv === Infinity) continue;
      piInv = div.pi === 0 ? Infinity : piInv + (coeffs[i] * coeffs[i]) / div.pi;
    }
    const pi = 1 / piInv;
    return target.updateMessage(this, new Gaussian(pi, pi * mu));
  }

  down() {
    return this._update(this.sum, this.terms, this.coeffs);
  }

  /** Solve the linear relation for terms[index] and send it that message. */
  up(index) {
    const c = this.coeffs[index];
    const coeffs = this.coeffs.map((ci, i) => (c === 0 ? 0 : i === index ? 1 / c : -ci / c));
    const vars = this.terms.slice();
    vars[index] = this.sum;
    return this._update(this.terms[index], vars, coeffs);
  }
}

/** The observation: a team difference is either > margin (win) or within the margin (draw). */
export class TruncateFactor extends Factor {
  constructor(variable, vFn, wFn, margin) {
    super([variable]);
    this.vFn = vFn;
    this.wFn = wFn;
    this.margin = margin;
  }

  up() {
    const v = this.vars[0];
    const div = v.value.div(v.message(this));
    const sqrtPi = Math.sqrt(div.pi);
    const t = div.tau / sqrtPi;
    const e = this.margin * sqrtPi;
    const vv = this.vFn(t, e);
    const ww = this.wFn(t, e);
    const denom = 1 - ww;
    return v.updateValue(this, new Gaussian(div.pi / denom, (div.tau + sqrtPi * vv) / denom));
  }
}
