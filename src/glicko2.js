// Glicko2 (Glickman, 2012 revision), with a composite opponent extension for
// team games. Each call to `rate` is treated as one rating period.

const SCALE = 173.7178;
const BASE = 1500;

const g = (phi) => 1 / Math.sqrt(1 + (3 * phi * phi) / (Math.PI * Math.PI));
const E = (mu, muJ, phiJ) => 1 / (1 + Math.exp(-g(phiJ) * (mu - muJ)));

export class Glicko2 {
  constructor({ rating = 1500, rd = 350, volatility = 0.06, tau = 0.5, epsilon = 1e-6 } = {}) {
    Object.assign(this, { rating, rd, volatility, tau, epsilon });
  }

  createRating(rating = this.rating, rd = this.rd, volatility = this.volatility) {
    return { rating, rd, volatility };
  }

  /** Step 5 of the paper: find the new volatility with the Illinois root finder. */
  _volatility(phi, sigma, delta, v) {
    const a = Math.log(sigma * sigma);
    const tau = this.tau;
    const f = (x) => {
      const ex = Math.exp(x);
      const d = phi * phi + v + ex;
      return (ex * (delta * delta - phi * phi - v - ex)) / (2 * d * d) - (x - a) / (tau * tau);
    };
    let A = a;
    let B;
    if (delta * delta > phi * phi + v) {
      B = Math.log(delta * delta - phi * phi - v);
    } else {
      let k = 1;
      while (f(a - k * tau) < 0) k++;
      B = a - k * tau;
    }
    let fA = f(A);
    let fB = f(B);
    for (let guard = 0; Math.abs(B - A) > this.epsilon && guard < 200; guard++) {
      const C = A + ((A - B) * fA) / (fB - fA);
      const fC = f(C);
      if (fC * fB <= 0) {
        A = B;
        fA = fB;
      } else {
        fA /= 2;
      }
      B = C;
      fB = fC;
    }
    return Math.exp(A / 2);
  }

  /**
   * Rate one player over a rating period.
   * @param player {rating, rd, volatility}
   * @param results [{opponent: {rating, rd}, score: 1 | 0.5 | 0}]
   */
  updatePlayer(player, results) {
    const mu = (player.rating - BASE) / SCALE;
    const phi = player.rd / SCALE;
    if (results.length === 0) {
      const phiStar = Math.sqrt(phi * phi + player.volatility ** 2);
      return { ...player, rd: phiStar * SCALE };
    }
    let vInv = 0;
    let sum = 0;
    for (const { opponent, score } of results) {
      const muJ = (opponent.rating - BASE) / SCALE;
      const phiJ = opponent.rd / SCALE;
      const e = E(mu, muJ, phiJ);
      vInv += g(phiJ) ** 2 * e * (1 - e);
      sum += g(phiJ) * (score - e);
    }
    const v = 1 / vInv;
    const delta = v * sum;
    const sigma = this._volatility(phi, player.volatility, delta, v);
    const phiStar = Math.sqrt(phi * phi + sigma * sigma);
    const phiNew = 1 / Math.sqrt(1 / (phiStar * phiStar) + 1 / v);
    const muNew = mu + phiNew * phiNew * sum;
    return { ...player, rating: muNew * SCALE + BASE, rd: phiNew * SCALE, volatility: sigma };
  }

  /** Glicko2 has no notion of teams, so a team plays as one composite opponent. */
  _composite(team) {
    const rating = team.reduce((s, p) => s + p.rating, 0) / team.length;
    const rd = Math.sqrt(team.reduce((s, p) => s + p.rd * p.rd, 0) / team.length);
    return { rating, rd };
  }

  winProbability(teamA, teamB) {
    const a = this._composite(teamA);
    const b = this._composite(teamB);
    const phi = Math.hypot(a.rd, b.rd) / SCALE;
    return E((a.rating - BASE) / SCALE, (b.rating - BASE) / SCALE, phi);
  }

  rate(teams, { ranks } = {}) {
    if (teams.length < 2) throw new RangeError('need at least two teams');
    ranks = ranks ?? teams.map((_, i) => i);
    const composites = teams.map((t) => this._composite(t));
    return teams.map((team, i) =>
      team.map((p) => {
        const results = [];
        for (let j = 0; j < teams.length; j++) {
          if (i === j) continue;
          const score = ranks[i] < ranks[j] ? 1 : ranks[i] === ranks[j] ? 0.5 : 0;
          results.push({ opponent: composites[j], score });
        }
        return this.updatePlayer(p, results);
      })
    );
  }
}
