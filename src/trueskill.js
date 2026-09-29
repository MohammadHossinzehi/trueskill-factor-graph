// TrueSkill (Herbrich, Minka and Graepel, NIPS 2006) implemented as a factor
// graph with expectation propagation. Handles any number of teams, any team
// sizes, draws, ties between some of the teams, and partial play weights.
import { Gaussian, cdf, ppf, vWin, wWin, vDraw, wDraw } from './gaussian.js';
import { Variable, PriorFactor, LikelihoodFactor, SumFactor, TruncateFactor } from './factorgraph.js';
import { transpose, multiply, add, scale, determinant, inverse } from './matrix.js';

export const DEFAULTS = Object.freeze({
  mu: 25,
  sigma: 25 / 3,
  beta: 25 / 6,
  tau: 25 / 300,
  drawProbability: 0.1,
  minDelta: 1e-4,
  maxIterations: 100,
});

export class Rating {
  constructor(mu = DEFAULTS.mu, sigma = DEFAULTS.sigma) {
    this.mu = mu;
    this.sigma = sigma;
  }

  /** Conservative skill estimate used for leaderboards: 99.7% sure the true skill is above this. */
  get exposure() {
    return this.mu - 3 * this.sigma;
  }

  toString() {
    return `Rating(mu=${this.mu.toFixed(3)}, sigma=${this.sigma.toFixed(3)})`;
  }
}

export class TrueSkill {
  constructor(options = {}) {
    Object.assign(this, DEFAULTS, options);
  }

  createRating(mu = this.mu, sigma = this.sigma) {
    return new Rating(mu, sigma);
  }

  /** Performance difference below which a game counts as a draw, for `size` players in total. */
  drawMargin(size) {
    return ppf((this.drawProbability + 1) / 2) * Math.sqrt(size) * this.beta;
  }

  _weights(teams, weights) {
    if (!weights) return teams.map((t) => t.map(() => 1));
    if (weights.length !== teams.length || weights.some((w, i) => w.length !== teams[i].length))
      throw new RangeError('weights must have the same shape as teams');
    // A zero weight would make the sum factor singular, so floor it at minDelta.
    return weights.map((w) => w.map((x) => Math.max(this.minDelta, x)));
  }

  /**
   * Update ratings after a match.
   * @param {Rating[][]} teams  teams of ratings, e.g. [[a], [b]] or [[a, b], [c, d]]
   * @param {{ranks?: number[], weights?: number[][]}} opts  lower rank is better, equal ranks are a draw
   * @returns {Rating[][]} new ratings, same shape and order as `teams`
   */
  rate(teams, { ranks, weights } = {}) {
    if (!Array.isArray(teams) || teams.length < 2) throw new RangeError('need at least two teams');
    if (teams.some((t) => !Array.isArray(t) || t.length === 0)) throw new RangeError('every team needs a player');
    ranks = ranks ?? teams.map((_, i) => i);
    if (ranks.length !== teams.length) throw new RangeError('one rank per team');
    const w = this._weights(teams, weights);

    // Sort teams best to worst; the schedule assumes neighbours are compared.
    const order = teams.map((_, i) => i).sort((a, b) => ranks[a] - ranks[b]);
    const sTeams = order.map((i) => teams[i]);
    const sRanks = order.map((i) => ranks[i]);
    const sWeights = order.map((i) => w[i]);

    const flat = sTeams.flat();
    const ratingVars = flat.map(() => new Variable('skill'));
    const perfVars = flat.map(() => new Variable('perf'));
    const teamPerfVars = sTeams.map(() => new Variable('teamPerf'));
    const diffVars = sTeams.slice(1).map(() => new Variable('diff'));

    const priors = flat.map(
      (r, i) => new PriorFactor(ratingVars[i], Gaussian.fromMuSigma(r.mu, r.sigma), this.tau)
    );
    const perfs = flat.map((_, i) => new LikelihoodFactor(ratingVars[i], perfVars[i], this.beta ** 2));

    let offset = 0;
    const teamPerfs = sTeams.map((team, t) => {
      const f = new SumFactor(teamPerfVars[t], perfVars.slice(offset, offset + team.length), sWeights[t]);
      offset += team.length;
      return f;
    });
    const diffs = diffVars.map(
      (v, i) => new SumFactor(v, [teamPerfVars[i], teamPerfVars[i + 1]], [1, -1])
    );
    const truncs = diffVars.map((v, i) => {
      const size = sTeams[i].length + sTeams[i + 1].length;
      const draw = sRanks[i] === sRanks[i + 1];
      const margin = this.drawMargin(size);
      if (draw && margin === 0) throw new RangeError('a draw is impossible when drawProbability is 0');
      return draw ? new TruncateFactor(v, vDraw, wDraw, margin) : new TruncateFactor(v, vWin, wWin, margin);
    });

    // Downward pass: priors -> performances -> team performances.
    for (const f of priors) f.down();
    for (const f of perfs) f.down();
    for (const f of teamPerfs) f.down();

    // The team differences form a chain whose links share team performances,
    // so sweep it forward and back until the messages stop moving.
    const n = diffs.length;
    this.lastIterations = 1;
    if (n === 1) {
      diffs[0].down();
      truncs[0].up();
    } else {
      for (let iter = 0; iter < this.maxIterations; iter++) {
        let d = 0;
        for (let x = 0; x < n - 1; x++) {
          diffs[x].down();
          d = Math.max(d, truncs[x].up());
          diffs[x].up(1);
        }
        for (let x = n - 1; x > 0; x--) {
          diffs[x].down();
          d = Math.max(d, truncs[x].up());
          diffs[x].up(0);
        }
        this.lastIterations = iter + 1;
        if (d <= this.minDelta) break;
      }
    }
    diffs[0].up(0);
    diffs[n - 1].up(1);

    // Upward pass back to the skills.
    for (const f of teamPerfs) for (let x = 0; x < f.terms.length; x++) f.up(x);
    for (const f of perfs) f.up();

    // Unflatten and restore the caller's team order.
    const result = new Array(teams.length);
    offset = 0;
    sTeams.forEach((team, t) => {
      result[order[t]] = team.map((_, j) => {
        const g = ratingVars[offset + j].value;
        return new Rating(g.mu, g.sigma);
      });
      offset += team.length;
    });
    return result;
  }

  /** Convenience wrapper for the common head to head case. */
  rate1vs1(winner, loser, drawn = false) {
    const [[a], [b]] = this.rate([[winner], [loser]], { ranks: drawn ? [0, 0] : [0, 1] });
    return [a, b];
  }

  /**
   * Match quality in [0, 1]: the probability of a draw relative to the
   * probability of a draw between two identical teams. Higher means fairer.
   */
  quality(teams, weights) {
    const w = this._weights(teams, weights);
    const flat = teams.flat();
    const mean = flat.map((r) => [r.mu]);
    const variance = flat.map((r, i) => flat.map((q, j) => (i === j ? r.sigma ** 2 : 0)));

    // A[player][pair] = +w if the player is in the first team of the pair, -w if in the second.
    const A = flat.map(() => new Array(teams.length - 1).fill(0));
    let offset = 0;
    const starts = teams.map((t) => {
      const s = offset;
      offset += t.length;
      return s;
    });
    for (let p = 0; p < teams.length - 1; p++) {
      teams[p].forEach((_, j) => (A[starts[p] + j][p] = w[p][j]));
      teams[p + 1].forEach((_, j) => (A[starts[p + 1] + j][p] = -w[p + 1][j]));
    }
    const At = transpose(A);
    const ata = scale(multiply(At, A), this.beta ** 2);
    const atsa = multiply(multiply(At, variance), A);
    const middle = add(ata, atsa);
    const start = transpose(multiply(At, mean));
    const end = multiply(At, mean);
    const eArg = -0.5 * multiply(multiply(start, inverse(middle)), end)[0][0];
    const sArg = determinant(ata) / determinant(middle);
    return Math.exp(eArg) * Math.sqrt(sArg);
  }

  _twoTeam(teamA, teamB) {
    const deltaMu = teamA.reduce((s, r) => s + r.mu, 0) - teamB.reduce((s, r) => s + r.mu, 0);
    const size = teamA.length + teamB.length;
    const sumSigma = [...teamA, ...teamB].reduce((s, r) => s + r.sigma ** 2, 0);
    const c = Math.sqrt(size * this.beta ** 2 + sumSigma);
    return { deltaMu, c, size };
  }

  /** Probability that teamA's performance exceeds teamB's (draw margin ignored). */
  winProbability(teamA, teamB) {
    const { deltaMu, c } = this._twoTeam(teamA, teamB);
    return cdf(deltaMu / c);
  }

  /** Full three way forecast for a two team match: {win, draw, loss} from teamA's side. */
  outcomeProbabilities(teamA, teamB) {
    const { deltaMu, c, size } = this._twoTeam(teamA, teamB);
    const e = this.drawMargin(size);
    const win = cdf((deltaMu - e) / c);
    const loss = cdf((-deltaMu - e) / c);
    return { win, draw: Math.max(0, 1 - win - loss), loss };
  }
}
