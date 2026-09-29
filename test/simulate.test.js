import { test } from 'node:test';
import assert from 'node:assert/strict';
import { generateLeague, mulberry32 } from '../src/simulate.js';
import { runBenchmark, spearman } from '../src/evaluate.js';

test('PRNG is deterministic and uniform enough', () => {
  const a = mulberry32(7);
  const b = mulberry32(7);
  const xs = Array.from({ length: 20000 }, () => a());
  assert.deepEqual(xs.slice(0, 5), Array.from({ length: 5 }, () => b()));
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
  assert.ok(Math.abs(mean - 0.5) < 0.01);
  assert.ok(xs.every((x) => x >= 0 && x < 1));
});

test('league generator respects the match format', () => {
  const { matches, skills } = generateLeague({ players: 30, matches: 200, mode: '2v2', seed: 3 });
  assert.equal(skills.length, 30);
  for (const { teams, ranks } of matches) {
    assert.equal(teams.length, 2);
    assert.ok(teams.every((t) => t.length === 2));
    assert.equal(new Set(teams.flat()).size, 4); // nobody plays against themselves
    assert.equal(ranks.length, 2);
  }
  assert.throws(() => generateLeague({ mode: 'nope' }), RangeError);
  assert.throws(() => generateLeague({ players: 3, mode: 'ffa4' }), RangeError);
});

test('simulated draw rate between equal players matches the target', () => {
  const { matches } = generateLeague({ players: 2, matches: 20000, skillSd: 0, drawProbability: 0.25, seed: 11 });
  const draws = matches.filter((m) => m.ranks[0] === m.ranks[1]).length / matches.length;
  assert.ok(Math.abs(draws - 0.25) < 0.015, `draw rate ${draws}`);
});

test('spearman handles ties and perfect orderings', () => {
  assert.equal(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1);
  assert.equal(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1);
  assert.ok(Math.abs(spearman([1, 1, 2, 3], [1, 2, 3, 4]) - 0.9486833) < 1e-6);
});

test('benchmark: every system beats a coin flip and none beats the oracle', () => {
  for (const mode of ['1v1', '2v2', 'ffa4']) {
    const { results } = runBenchmark({ mode, players: 60, matches: 1500, seed: 5 });
    const oracle = results.find((r) => r.system.startsWith('Oracle'));
    for (const r of results) {
      assert.ok(r.logLoss < Math.LN2, `${r.system} ${mode} log loss ${r.logLoss}`);
      assert.ok(r.logLoss >= oracle.logLoss - 0.02, `${r.system} ${mode} beat the oracle?`);
      assert.ok(r.spearman > 0.8, `${r.system} ${mode} spearman ${r.spearman}`);
    }
  }
});

test('benchmark: TrueSkill is the best calibrated system in team games', () => {
  const { results } = runBenchmark({ mode: '2v2', players: 100, matches: 3000, seed: 9 });
  const by = Object.fromEntries(results.map((r) => [r.system, r]));
  assert.ok(by.TrueSkill.logLoss < by['Elo (K=24)'].logLoss);
  assert.ok(by.TrueSkill.logLoss < by.Glicko2.logLoss);
});
