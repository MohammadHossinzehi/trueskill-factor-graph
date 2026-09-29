import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Elo } from '../src/elo.js';
import { Glicko2 } from '../src/glicko2.js';

const close = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} != ${b}`);

test('Glicko2 reproduces the worked example from Glickman\'s paper', () => {
  const gl = new Glicko2({ tau: 0.5 });
  const out = gl.updatePlayer({ rating: 1500, rd: 200, volatility: 0.06 }, [
    { opponent: { rating: 1400, rd: 30 }, score: 1 },
    { opponent: { rating: 1550, rd: 100 }, score: 0 },
    { opponent: { rating: 1700, rd: 300 }, score: 0 },
  ]);
  // Paper: r' = 1464.06, RD' = 151.52, sigma' = 0.05999 (it rounds intermediates).
  close(out.rating, 1464.06, 0.02);
  close(out.rd, 151.52, 0.01);
  close(out.volatility, 0.05999, 1e-5);
});

test('Glicko2 inactivity only grows RD', () => {
  const gl = new Glicko2();
  const p = { rating: 1700, rd: 50, volatility: 0.06 };
  const q = gl.updatePlayer(p, []);
  assert.equal(q.rating, 1700);
  assert.ok(q.rd > 50);
});

test('Glicko2 team game against a composite opponent', () => {
  const gl = new Glicko2();
  const [[a, b], [c]] = gl.rate([[gl.createRating(), gl.createRating()], [gl.createRating()]]);
  assert.ok(a.rating > 1500 && b.rating > 1500 && c.rating < 1500);
  close(gl.winProbability([a, b], [c]) + gl.winProbability([c], [a, b]), 1, 1e-12);
});

test('Elo is zero sum between two equal size teams', () => {
  const elo = new Elo({ k: 32 });
  const [[a], [b]] = elo.rate([[{ rating: 1600 }], [{ rating: 1400 }]], { ranks: [1, 0] });
  close(a.rating + b.rating, 3000, 1e-9);
  // Upset: the favourite loses 32 * 0.7597 points.
  close(1600 - a.rating, 32 * (1 / (1 + 10 ** (-200 / 400))), 1e-9);
});

test('Elo draw between equals changes nothing; free for all splits K', () => {
  const elo = new Elo({ k: 30 });
  const [[a]] = elo.rate([[{ rating: 1500 }], [{ rating: 1500 }]], { ranks: [0, 0] });
  assert.equal(a.rating, 1500);
  const ffa = elo.rate([[{ rating: 1500 }], [{ rating: 1500 }], [{ rating: 1500 }]]).flat();
  close(ffa[0].rating, 1515, 1e-9);
  close(ffa[1].rating, 1500, 1e-9);
  close(ffa[2].rating, 1485, 1e-9);
});
