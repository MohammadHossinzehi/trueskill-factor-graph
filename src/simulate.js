// Synthetic leagues with a known ground truth. Every player has a hidden true
// skill; each match draws a noisy performance per player and the outcome
// follows from those performances. Because we know the truth we can score
// how fast and how well each rating system recovers it.
import { ppf } from './gaussian.js';

/** Small, fast, seedable PRNG (mulberry32). Returns floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sampler (Box Muller) on top of a uniform source. */
export function normal(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export const MODES = Object.freeze({
  '1v1': { teams: 2, size: 1 },
  '2v2': { teams: 2, size: 2 },
  '3v3': { teams: 2, size: 3 },
  ffa4: { teams: 4, size: 1 },
  ffa8: { teams: 8, size: 1 },
});

/**
 * Build a league: true skills plus a stream of matches with outcomes.
 * @returns {{skills: number[], matches: {teams: number[][], ranks: number[]}[]}}
 */
export function generateLeague({
  players = 200,
  matches = 5000,
  mode = '1v1',
  seed = 1,
  skillMean = 25,
  skillSd = 25 / 3,
  beta = 25 / 6,
  drawProbability = 0.1,
} = {}) {
  const shape = MODES[mode];
  if (!shape) throw new RangeError(`unknown mode ${mode}; try one of ${Object.keys(MODES).join(', ')}`);
  const need = shape.teams * shape.size;
  if (players < need) throw new RangeError(`mode ${mode} needs at least ${need} players`);
  const rand = mulberry32(seed);
  const skills = Array.from({ length: players }, () => skillMean + skillSd * normal(rand));

  // Only two team games can be drawn; the margin is calibrated so that equally
  // skilled teams draw with probability `drawProbability`.
  const margin = ppf((drawProbability + 1) / 2) * Math.sqrt(need) * beta;

  const out = [];
  const ids = Array.from({ length: players }, (_, i) => i);
  for (let m = 0; m < matches; m++) {
    // Partial Fisher Yates shuffle: pick `need` distinct random players.
    for (let i = 0; i < need; i++) {
      const j = i + Math.floor(rand() * (players - i));
      [ids[i], ids[j]] = [ids[j], ids[i]];
    }
    const teams = [];
    for (let t = 0; t < shape.teams; t++) teams.push(ids.slice(t * shape.size, (t + 1) * shape.size));
    const perf = teams.map((team) => team.reduce((s, p) => s + skills[p] + beta * normal(rand), 0));

    let ranks;
    if (shape.teams === 2) {
      const d = perf[0] - perf[1];
      ranks = Math.abs(d) <= margin ? [0, 0] : d > 0 ? [0, 1] : [1, 0];
    } else {
      const order = perf.map((_, i) => i).sort((a, b) => perf[b] - perf[a]);
      ranks = new Array(shape.teams);
      order.forEach((team, place) => (ranks[team] = place));
    }
    out.push({ teams, ranks });
  }
  return { skills, matches: out, margin };
}
