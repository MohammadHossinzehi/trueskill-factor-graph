// Head to head benchmark of the rating systems on a synthetic league.
//
// Every match is scored *before* the system sees its result (prequential
// evaluation), so the metrics measure genuine out of sample prediction.
import { TrueSkill } from './trueskill.js';
import { Elo } from './elo.js';
import { Glicko2 } from './glicko2.js';
import { generateLeague } from './simulate.js';

/** Adapters give every system the same tiny interface. */
export function defaultSystems() {
  const ts = new TrueSkill();
  const elo = new Elo();
  const gl = new Glicko2();
  return [
    {
      name: 'TrueSkill',
      init: () => ts.createRating(),
      // Draws are excluded from scoring, so condition on a decisive result.
      predict: (a, b) => {
        const p = ts.outcomeProbabilities(a, b);
        return p.win / (p.win + p.loss);
      },
      rate: (teams, ranks) => ts.rate(teams, { ranks }),
      point: (r) => r.mu,
    },
    {
      name: 'Glicko2',
      init: () => gl.createRating(),
      predict: (a, b) => gl.winProbability(a, b),
      rate: (teams, ranks) => gl.rate(teams, { ranks }),
      point: (r) => r.rating,
    },
    {
      name: 'Elo (K=24)',
      init: () => elo.createRating(),
      predict: (a, b) => elo.winProbability(a, b),
      rate: (teams, ranks) => elo.rate(teams, { ranks }),
      point: (r) => r.rating,
    },
  ];
}

/** Average ranks (ties share the mean rank), used for Spearman correlation. */
function rankdata(xs) {
  const idx = xs.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const ranks = new Array(xs.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k][1]] = r;
    i = j + 1;
  }
  return ranks;
}

export function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  return sxx === 0 || syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy);
}

export const spearman = (xs, ys) => pearson(rankdata(xs), rankdata(ys));

/**
 * Run every system over the same league.
 * @param opts  league options (see generateLeague) plus `warmup` in [0, 1):
 *              the leading fraction of matches excluded from the prediction metrics.
 */
export function runBenchmark({ warmup = 0.2, systems = defaultSystems(), ...leagueOpts } = {}) {
  const league = generateLeague(leagueOpts);
  const { skills, matches } = league;
  const scoreFrom = Math.floor(matches.length * warmup);
  const eps = 1e-12;

  const results = systems.map((sys) => {
    const ratings = skills.map(() => sys.init());
    let logLoss = 0;
    let brier = 0;
    let correct = 0;
    let scored = 0;
    const t0 = performance.now();
    matches.forEach(({ teams, ranks }, m) => {
      const teamRatings = teams.map((team) => team.map((p) => ratings[p]));
      if (m >= scoreFrom) {
        for (let i = 0; i < teams.length; i++)
          for (let j = i + 1; j < teams.length; j++) {
            if (ranks[i] === ranks[j]) continue;
            const p = Math.min(1 - eps, Math.max(eps, sys.predict(teamRatings[i], teamRatings[j])));
            const y = ranks[i] < ranks[j] ? 1 : 0;
            logLoss -= y ? Math.log(p) : Math.log(1 - p);
            brier += (p - y) ** 2;
            correct += (p > 0.5) === (y === 1) ? 1 : 0;
            scored++;
          }
      }
      const updated = sys.rate(teamRatings, ranks);
      teams.forEach((team, t) => team.forEach((p, k) => (ratings[p] = updated[t][k])));
    });
    const ms = performance.now() - t0;
    return {
      system: sys.name,
      logLoss: logLoss / scored,
      brier: brier / scored,
      accuracy: correct / scored,
      spearman: spearman(ratings.map(sys.point), skills),
      comparisons: scored,
      ms,
    };
  });

  // What a perfect oracle that knows the true skills could achieve: the floor
  // set by irreducible performance noise.
  const beta = leagueOpts.beta ?? 25 / 6;
  const oracle = new TrueSkill({ beta });
  let oLoss = 0;
  let oScored = 0;
  let oCorrect = 0;
  let oBrier = 0;
  matches.slice(scoreFrom).forEach(({ teams, ranks }) => {
    const tr = teams.map((team) => team.map((p) => ({ mu: skills[p], sigma: 1e-9 })));
    for (let i = 0; i < teams.length; i++)
      for (let j = i + 1; j < teams.length; j++) {
        if (ranks[i] === ranks[j]) continue;
        const pr = oracle.outcomeProbabilities(tr[i], tr[j]);
        let p = teams.length === 2 ? pr.win / (pr.win + pr.loss) : oracle.winProbability(tr[i], tr[j]);
        p = Math.min(1 - eps, Math.max(eps, p));
        const y = ranks[i] < ranks[j] ? 1 : 0;
        oLoss -= y ? Math.log(p) : Math.log(1 - p);
        oBrier += (p - y) ** 2;
        oCorrect += (p > 0.5) === (y === 1) ? 1 : 0;
        oScored++;
      }
  });
  results.push({
    system: 'Oracle (true skill)',
    logLoss: oLoss / oScored,
    brier: oBrier / oScored,
    accuracy: oCorrect / oScored,
    spearman: 1,
    comparisons: oScored,
    ms: 0,
  });
  return { results, league };
}
