#!/usr/bin/env node
// Command line benchmark: pits TrueSkill, Glicko2 and Elo against each other
// on a simulated league where the true skills are known.
import { runBenchmark } from '../src/evaluate.js';
import { MODES } from '../src/simulate.js';

const HELP = `usage: ratingbench [options]

  --mode <m>       match format: ${Object.keys(MODES).join(', ')}   (default 1v1)
  --players <n>    league size                                (default 200)
  --matches <n>    number of matches                          (default 5000)
  --seed <n>       RNG seed, runs are fully reproducible      (default 1)
  --warmup <f>     fraction of early matches not scored       (default 0.2)
  --json           print machine readable JSON instead of a table
  --help           show this message`;

function parseArgs(argv) {
  const opts = { mode: '1v1', players: 200, matches: 5000, seed: 1, warmup: 0.2, json: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--help' || a === '-h') {
      console.log(HELP);
      process.exit(0);
    } else if (a === '--json') opts.json = true;
    else if (a.startsWith('--') && a.slice(2) in opts) {
      const key = a.slice(2);
      const val = argv[++i];
      if (val === undefined) throw new Error(`missing value for ${a}`);
      opts[key] = key === 'mode' ? val : Number(val);
      if (key !== 'mode' && !Number.isFinite(opts[key])) throw new Error(`${a} expects a number`);
    } else throw new Error(`unknown argument ${a}`);
  }
  return opts;
}

try {
  const { json, ...opts } = parseArgs(process.argv.slice(2));
  const { results } = runBenchmark(opts);
  if (json) {
    console.log(JSON.stringify({ options: opts, results }, null, 2));
  } else {
    console.log(
      `league: ${opts.players} players, ${opts.matches} ${opts.mode} matches, seed ${opts.seed}, ` +
        `first ${Math.round(opts.warmup * 100)}% unscored\n`
    );
    const header = ['system', 'log loss', 'brier', 'accuracy', 'spearman', 'time'];
    const rows = results.map((r) => [
      r.system,
      r.logLoss.toFixed(4),
      r.brier.toFixed(4),
      (r.accuracy * 100).toFixed(1) + '%',
      r.spearman.toFixed(3),
      r.ms ? r.ms.toFixed(0) + ' ms' : '',
    ]);
    const widths = header.map((h, c) => Math.max(h.length, ...rows.map((r) => r[c].length)));
    const line = (cells) => cells.map((x, c) => (c === 0 ? x.padEnd(widths[c]) : x.padStart(widths[c]))).join('  ');
    console.log(line(header));
    console.log(widths.map((w) => '='.repeat(w)).join('  '));
    rows.forEach((r) => console.log(line(r)));
    console.log(`\ncoin flip log loss is ${Math.LN2.toFixed(4)}; lower is better, the oracle is the floor.`);
  }
} catch (err) {
  console.error(`ratingbench: ${err.message}\n\n${HELP}`);
  process.exit(1);
}
