// Classic Elo, generalised to teams (team strength = mean member rating) and
// free for all games (decomposed into every pair of teams).

export class Elo {
  constructor({ initial = 1500, k = 24, scale = 400 } = {}) {
    this.initial = initial;
    this.k = k;
    this.scale = scale;
  }

  createRating(rating = this.initial) {
    return { rating };
  }

  expected(ra, rb) {
    return 1 / (1 + 10 ** ((rb - ra) / this.scale));
  }

  _strength(team) {
    return team.reduce((s, p) => s + p.rating, 0) / team.length;
  }

  winProbability(teamA, teamB) {
    return this.expected(this._strength(teamA), this._strength(teamB));
  }

  rate(teams, { ranks } = {}) {
    if (teams.length < 2) throw new RangeError('need at least two teams');
    ranks = ranks ?? teams.map((_, i) => i);
    const strength = teams.map((t) => this._strength(t));
    const k = this.k / (teams.length - 1);
    const delta = teams.map(() => 0);
    for (let i = 0; i < teams.length; i++)
      for (let j = 0; j < teams.length; j++) {
        if (i === j) continue;
        const score = ranks[i] < ranks[j] ? 1 : ranks[i] === ranks[j] ? 0.5 : 0;
        delta[i] += k * (score - this.expected(strength[i], strength[j]));
      }
    return teams.map((t, i) => t.map((p) => ({ ...p, rating: p.rating + delta[i] })));
  }
}
