import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TrueSkill, Rating } from '../src/trueskill.js';
import { vWin, wWin, vDraw, wDraw } from '../src/gaussian.js';

const close = (a, b, tol = 1e-3) => assert.ok(Math.abs(a - b) <= tol, `${a} != ${b}`);
const ts = new TrueSkill();
const fresh = () => ts.createRating();

// Reference numbers are the published values of Microsoft's TrueSkill
// calculator, as reproduced by the Moserware and trueskill.py test suites.
test('1 vs 1, decisive', () => {
  const [w, l] = ts.rate1vs1(fresh(), fresh());
  close(w.mu, 29.396); close(w.sigma, 7.171);
  close(l.mu, 20.604); close(l.sigma, 7.171);
});

test('1 vs 1, draw', () => {
  const [a, b] = ts.rate1vs1(fresh(), fresh(), true);
  close(a.mu, 25); close(a.sigma, 6.458);
  close(b.mu, 25); close(b.sigma, 6.458);
});

test('three player free for all', () => {
  const out = ts.rate([[fresh()], [fresh()], [fresh()]]).flat();
  const expected = [[31.675, 6.656], [25.0, 6.208], [18.325, 6.656]];
  out.forEach((r, i) => { close(r.mu, expected[i][0]); close(r.sigma, expected[i][1]); });
});

test('four player free for all', () => {
  const out = ts.rate([[fresh()], [fresh()], [fresh()], [fresh()]]).flat();
  const expected = [[33.207, 6.348], [27.401, 5.787], [22.599, 5.787], [16.793, 6.348]];
  out.forEach((r, i) => { close(r.mu, expected[i][0]); close(r.sigma, expected[i][1]); });
});

test('2 vs 2', () => {
  const [a, b] = ts.rate([[fresh(), fresh()], [fresh(), fresh()]]);
  for (const r of a) { close(r.mu, 28.108); close(r.sigma, 7.774); }
  for (const r of b) { close(r.mu, 21.892); close(r.sigma, 7.774); }
});

test('match quality of evenly matched games', () => {
  close(ts.quality([[fresh()], [fresh()]]), 0.447);
  close(ts.quality([[fresh(), fresh()], [fresh(), fresh()]]), 0.447);
  // A lopsided match is lower quality.
  assert.ok(ts.quality([[new Rating(40, 2)], [new Rating(10, 2)]]) < 0.01);
});

// Independent check: for two teams the factor graph collapses to the closed
// form update from the TrueSkill paper, so both code paths must agree.
function closedForm(env, winners, losers, drawn) {
  const all = [...winners, ...losers];
  const s2 = (r) => r.sigma ** 2 + env.tau ** 2;
  const c = Math.sqrt(all.reduce((s, r) => s + s2(r), 0) + all.length * env.beta ** 2);
  const d = winners.reduce((s, r) => s + r.mu, 0) - losers.reduce((s, r) => s + r.mu, 0);
  const e = env.drawMargin(all.length) / c;
  const v = drawn ? vDraw(d / c, e) : vWin(d / c, e);
  const w = drawn ? wDraw(d / c, e) : wWin(d / c, e);
  const upd = (r, sign) => ({
    mu: r.mu + sign * (s2(r) / c) * v,
    sigma: Math.sqrt(s2(r) * (1 - (s2(r) / (c * c)) * w)),
  });
  return [winners.map((r) => upd(r, 1)), losers.map((r) => upd(r, -1))];
}

test('two team updates agree with the closed form equations', () => {
  const cases = [
    [[new Rating(30, 4)], [new Rating(22, 6)]],
    [[new Rating(18, 3)], [new Rating(35, 2)]], // upset
    [[new Rating(25, 8), new Rating(31, 1.5)], [new Rating(40, 5)]], // uneven team sizes
    [[new Rating(20, 2), new Rating(22, 3), new Rating(24, 4)], [new Rating(26, 5), new Rating(21, 1)]],
  ];
  for (const [a, b] of cases)
    for (const drawn of [false, true]) {
      const got = ts.rate([a, b], { ranks: drawn ? [0, 0] : [0, 1] });
      const want = closedForm(ts, a, b, drawn);
      got.flat().forEach((r, i) => {
        close(r.mu, want.flat()[i].mu, 1e-9);
        close(r.sigma, want.flat()[i].sigma, 1e-9);
      });
    }
});

test('ranks may be given in any order and results come back in input order', () => {
  const a = new Rating(20, 5), b = new Rating(30, 5), c = new Rating(25, 5);
  const sorted = ts.rate([[b], [c], [a]]);
  const shuffled = ts.rate([[a], [b], [c]], { ranks: [2, 0, 1] });
  close(shuffled[0][0].mu, sorted[2][0].mu, 1e-12);
  close(shuffled[1][0].mu, sorted[0][0].mu, 1e-12);
  close(shuffled[2][0].mu, sorted[1][0].mu, 1e-12);
});

test('partial draws inside a free for all', () => {
  const [[a], [b], [c]] = ts.rate([[fresh()], [fresh()], [fresh()]], { ranks: [0, 1, 1] });
  // A beat B directly and C only through the draw, so B and C end up close
  // but not identical: the graph is a chain, not a symmetric structure.
  assert.ok(a.mu > b.mu && a.mu > c.mu);
  assert.ok(b.mu < 25 && c.mu < 25);
  close(b.mu, c.mu, 0.05);
});

test('sigma always shrinks when tau is zero, and uncertain players move more', () => {
  const env = new TrueSkill({ tau: 0 });
  const vet = new Rating(25, 1);
  const rookie = new Rating(25, 8);
  const [[v], [r]] = env.rate([[vet], [rookie]]);
  assert.ok(v.sigma < vet.sigma && r.sigma < rookie.sigma);
  assert.ok(Math.abs(r.mu - 25) > 10 * Math.abs(v.mu - 25));
});

test('partial play weights scale how much a player is credited', () => {
  const [[full, half]] = ts.rate([[fresh(), fresh()], [fresh(), fresh()]], {
    weights: [[1, 0.5], [1, 1]],
  });
  assert.ok(full.mu > half.mu && half.mu > 25);
  // A weight of zero leaves the player (almost) untouched apart from tau drift.
  const [[, ghost]] = ts.rate([[fresh(), fresh()], [fresh()]], { weights: [[1, 0], [1]] });
  close(ghost.mu, 25, 1e-2);
});

test('win probabilities are symmetric and consistent', () => {
  const a = [new Rating(30, 3)];
  const b = [new Rating(24, 4)];
  close(ts.winProbability(a, b) + ts.winProbability(b, a), 1, 1e-12);
  const p = ts.outcomeProbabilities(a, b);
  close(p.win + p.draw + p.loss, 1, 1e-12);
  assert.ok(p.win > p.loss);
  const even = ts.outcomeProbabilities([fresh()], [fresh()]);
  close(even.win, even.loss, 1e-12);
});

test('draw probability of equal players matches the configured rate when sigma is 0', () => {
  const env = new TrueSkill({ drawProbability: 0.2 });
  const p = env.outcomeProbabilities([new Rating(25, 1e-9)], [new Rating(25, 1e-9)]);
  close(p.draw, 0.2, 1e-6);
});

test('input validation', () => {
  assert.throws(() => ts.rate([[fresh()]]), RangeError);
  assert.throws(() => ts.rate([[fresh()], []]), RangeError);
  assert.throws(() => ts.rate([[fresh()], [fresh()]], { ranks: [0] }), RangeError);
  assert.throws(() => ts.rate([[fresh()], [fresh()]], { weights: [[1]] }), RangeError);
  const noDraws = new TrueSkill({ drawProbability: 0 });
  assert.throws(() => noDraws.rate([[fresh()], [fresh()]], { ranks: [0, 0] }), RangeError);
});

test('the iterative schedule converges for large free for alls', () => {
  const env = new TrueSkill();
  const field = Array.from({ length: 16 }, (_, i) => [new Rating(20 + i, 3 + (i % 4))]);
  const out = env.rate(field).flat();
  assert.ok(env.lastIterations < env.maxIterations);
  out.forEach((r) => assert.ok(Number.isFinite(r.mu) && r.sigma > 0));
});
