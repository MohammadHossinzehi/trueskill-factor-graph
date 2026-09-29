# trueskill factor graph

TrueSkill implemented from the paper as an actual factor graph with expectation propagation, in plain JavaScript with zero dependencies. Next to it sit Glicko2 and Elo, and a simulated league with known ground truth skills so the three systems can be compared on equal terms instead of on vibes.

```
league: 200 players, 5000 2v2 matches, seed 1, first 20% unscored

system               log loss   brier  accuracy  spearman    time
===================  ========  ======  ========  ========  ======
TrueSkill              0.3339  0.1067     84.5%     0.983  159 ms
Glicko2                0.4705  0.1479     83.3%     0.965   34 ms
Elo (K=24)             0.4578  0.1437     83.3%     0.976   21 ms
Oracle (true skill)    0.2913  0.0929     86.7%     1.000
```

## Why this exists

Most "TrueSkill in N lines" snippets only handle 1v1 games using the closed form update. That hides the interesting part. The real model is a Bayesian network: every player has a Gaussian skill, every match produces a noisy performance per player, team performance is the sum of its players, and the observed ranking constrains the differences between neighbouring teams. Once you have more than two teams those constraints interact, the posterior is no longer Gaussian, and you need approximate inference to get a rating update at all.

This repo builds that graph explicitly (variables, prior, likelihood, sum and truncation factors) and runs the message schedule from Herbrich, Minka and Graepel (2006). As a result it handles, with one code path:

* any number of teams and any team sizes (1v1, 2v3, 16 player free for all ...)
* draws, including partial draws such as two players tied for second in a free for all
* partial play weights (a player who was only in the game half the time gets half the credit)
* match quality, the "how fair is this match" number used for matchmaking
* win, draw and loss probabilities for forecasting

Glicko2 and Elo are included because a rating system is only interesting relative to the alternatives, and the benchmark shows where the extra machinery pays off: in team games TrueSkill's log loss is dramatically better, because it reasons about each individual's uncertainty inside a team instead of treating the team as one blob.

## Running it

Requires Node 18 or newer. There is nothing to install.

```bash
git clone https://github.com/MohammadHossinzehi/trueskill-factor-graph
cd trueskill-factor-graph
npm test                                  # 33 tests, about a second
npm run bench                             # 1v1 league with the defaults
node bin/bench.js --mode 2v2              # team games
node bin/bench.js --mode ffa4 --seed 7    # four player free for all
node bin/bench.js --mode 3v3 --json       # machine readable output
```

Benchmark options: `--mode` (1v1, 2v2, 3v3, ffa4, ffa8), `--players`, `--matches`, `--seed`, `--warmup` (fraction of early matches that are not scored) and `--json`.

## Using the library

```js
import { TrueSkill } from './src/index.js';

const ts = new TrueSkill(); // mu 25, sigma 25/3, beta 25/6, tau 25/300, 10% draws

let alice = ts.createRating(), bob = ts.createRating();
let cara = ts.createRating(), dan = ts.createRating();

// Alice and Bob beat Cara and Dan (lower rank is better; ranks default to input order).
[[alice, bob], [cara, dan]] = ts.rate([[alice, bob], [cara, dan]]);

// Then Alice and Cara draw a 1v1.
[[alice], [cara]] = ts.rate([[alice], [cara]], { ranks: [0, 0] });

alice.mu;        // 25.696
alice.sigma;     // 6.082
alice.exposure;  // 7.449, the conservative mu minus 3 sigma used for leaderboards

ts.quality([[alice], [cara]]);             // 0.560, higher means a fairer match
ts.outcomeProbabilities([alice], [cara]);  // { win: 0.525, draw: 0.056, loss: 0.419 }

// Free for all with a tie for second, and a substitute who only played half the game:
ts.rate([[p1], [p2], [p3]], { ranks: [0, 1, 1] });
ts.rate([[p1, sub], [p2, p3]], { weights: [[1, 0.5], [1, 1]] });
```

`Glicko2` and `Elo` expose the same `createRating`, `rate(teams, { ranks })` and `winProbability(teamA, teamB)` shape, so they can be swapped into the benchmark or your own code.

## How it is put together

| file | what it does |
| --- | --- |
| `src/gaussian.js` | Gaussians in natural parameters (precision and precision adjusted mean), erfc and its inverse, and the v and w truncation corrections for wins and draws |
| `src/factorgraph.js` | `Variable` plus the four factor types. Each variable keeps the last message from every neighbouring factor so a message can be swapped without recomputing the whole product |
| `src/trueskill.js` | builds the graph for a match, runs the schedule, reads the posteriors back out; also match quality and outcome probabilities |
| `src/matrix.js` | the few dense matrix operations the quality formula needs |
| `src/glicko2.js` | Glicko2 with the Illinois volatility solver, plus a composite opponent extension for teams |
| `src/elo.js` | Elo with mean team strength and pairwise decomposition for free for alls |
| `src/simulate.js` | seeded league generator: hidden skills, noisy performances, calibrated draw margin |
| `src/evaluate.js` | prequential benchmark and the Spearman correlation it reports |
| `bin/bench.js` | command line front end |

### Design decisions

**Natural parameters everywhere.** Storing `pi = 1/sigma^2` and `tau = mu * pi` turns multiplying and dividing Gaussians into addition and subtraction, which is what message passing does all day. A message with `pi = 0` is the uniform "I know nothing" message, which makes initialisation trivial.

**The schedule.** Priors, performances and team sums are computed once on the way down. The team difference factors form a chain, and with three or more teams each link's truncation changes what its neighbours believe, so the chain is swept forward and back until no message moves by more than `minDelta` (a handful of sweeps in practice). Then messages flow back up to the skills. Teams are sorted by rank internally and results are returned in the caller's order.

**Numerical care in the tails.** A huge upset (a 5 sigma underdog winning) asks for `pdf(x) / cdf(x)` where `cdf(x)` underflows. The erfc approximation used is relatively accurate deep in the tail, `v` falls back to its asymptote `-x` when needed, and `w` is clamped strictly inside (0, 1) so the variance update can never divide by zero or go negative. There is a test that throws a 40 sigma upset at it.

**Zero weights** would make a sum factor singular, so partial play weights are floored at `minDelta`; a weight of zero therefore leaves a player effectively untouched.

**Evaluation is prequential.** Every match is predicted before the system sees its result, so the log loss really is out of sample. Draws are left out of scoring and TrueSkill's forecast is conditioned on a decisive result, which keeps the comparison fair to Elo and Glicko2, which have no explicit draw model. The oracle row predicts with the true skills, so it marks the floor set by performance noise: no system can beat it except by luck.

**Honest baselines.** Elo uses a standard K of 24 and is not tuned per league, and Glicko2 treats each match as its own rating period with a composite opponent for teams. Both are reasonable real world defaults, not strawmen. In 1v1 Glicko2 is close to TrueSkill, which is what you would expect since both track uncertainty; the gap opens up in team games.

## Testing

`npm test` runs 33 tests with the built in `node:test` runner:

* **Reference values.** 1v1 win and draw, 3 and 4 player free for alls, 2v2, and match quality all match the published TrueSkill calculator numbers to three decimals. Glicko2 reproduces the worked example in Glickman's paper (1464.06, 151.52, 0.05999).
* **An independent code path.** For two teams the factor graph must agree with the closed form update equations from the paper. The tests implement those equations separately and check agreement to 1e-9 across upsets, uneven team sizes and draws.
* **The maths underneath.** The v and w corrections are checked against brute force numerical moments of a truncated normal; cdf, ppf and erfc against known values, including the far tail.
* **Behaviour.** Result ordering, partial draws, partial play weights, sigma shrinking, symmetric probabilities, draw rate calibration, input validation and convergence on a 16 player free for all.
* **The benchmark itself.** Every system beats a coin flip, nobody beats the oracle, and TrueSkill wins on calibration in team games.

## References

* R. Herbrich, T. Minka, T. Graepel. *TrueSkill: A Bayesian Skill Rating System.* NIPS 2006.
* J. Moser. *Computing Your Skill.* 2010, the clearest walkthrough of the factor graph schedule.
* M. Glickman. *Example of the Glicko2 system.* 2012 revision.
