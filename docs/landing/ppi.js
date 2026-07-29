export const Z_SCORE_95 = 1.959963984540054;

// Fréchet bounds: a covariance between two Bernoulli(p), Bernoulli(q) variables is feasible
// only within [max(0, p + q - 1) - p*q, min(p, q) - p*q]. Clamp so impossible correlations
// (e.g. rho = 1 with p != q) degrade gracefully instead of producing a negative variance.
function clampCovariance(rawCovariance, trueMean, proxyMean) {
  const lowerCovariance = Math.max(0, trueMean + proxyMean - 1) - trueMean * proxyMean;
  const upperCovariance = Math.min(trueMean, proxyMean) - trueMean * proxyMean;
  const clamped = Math.min(Math.max(rawCovariance, lowerCovariance), upperCovariance);
  return clamped;
}

/**
 * Simulate the three estimators (human-only, judge-only, PPI) from population parameters.
 *
 * @param {Object} params
 * @param {number} params.totalSize      Total proxy-labeled dataset size N (e.g. 3300).
 * @param {number} params.humanSize      Human-labeled subset size n, 0 < n < N (e.g. 265).
 * @param {number} params.trueMean       Human accuracy p in (0, 1).
 * @param {number} params.proxyMean      Judge accuracy q in (0, 1).
 * @param {number} params.correlation    Human/judge correlation rho in [-1, 1].
 * @returns {{humanOnly: {mean: number, halfWidth: number}, judgeOnly: {mean: number, halfWidth: number}, ppi: {mean: number, halfWidth: number}, effectiveSampleSize: number, powerMultiplier: number}}
 */
export function simulate({ totalSize, humanSize, trueMean, proxyMean, correlation }) {
  if (!(humanSize > 0 && humanSize < totalSize)) {
    throw new Error(`'humanSize' must satisfy 0 < humanSize < totalSize; got ${humanSize}.`);
  }
  for (const [name, value] of [["trueMean", trueMean], ["proxyMean", proxyMean]]) {
    if (!(value > 0 && value < 1)) {
      throw new Error(`'${name}' must be in (0, 1); got ${value}.`);
    }
  }

  const numberUnlabeled = totalSize - humanSize;
  const trueVariance = trueMean * (1 - trueMean);
  const proxyVariance = proxyMean * (1 - proxyMean);
  const rawCovariance = correlation * Math.sqrt(trueVariance * proxyVariance);
  const labeledCovariance = clampCovariance(rawCovariance, trueMean, proxyMean);

  // Power-tuned lambda (PPI++) with population moments: lambda* = cov / ((1 + n/N) * Var(f)).
  const factor = 1 + humanSize / numberUnlabeled;
  const tuningParameter = labeledCovariance / (factor * proxyVariance);

  // PPI standard error: the asymptotic limit of glide's _compute_std_estimate with population
  // variances. Var(Y - lambda*f) = Var(Y) - 2*lambda*Cov + lambda^2 * Var(f).
  const residualVariance = trueVariance - 2 * tuningParameter * labeledCovariance + tuningParameter ** 2 * proxyVariance;
  const ppiVariance = residualVariance / humanSize + (tuningParameter ** 2 * proxyVariance) / numberUnlabeled;
  const ppiStd = Math.sqrt(ppiVariance);

  // Human-only (classical) and judge-only (naive: treats every proxy label as ground truth).
  const humanStd = Math.sqrt(trueVariance / humanSize);
  const judgeStd = Math.sqrt(proxyVariance / totalSize);

  // Effective sample size, mirroring glide: floor(n * Var_classical / Var_ppi),
  // with Var_classical = Var(Y)/n, hence floor(Var(Y) / Var_ppi).
  const effectiveSampleSize = Math.floor(trueVariance / ppiVariance);

  const result = {
    humanOnly: { mean: trueMean, halfWidth: Z_SCORE_95 * humanStd },
    judgeOnly: { mean: proxyMean, halfWidth: Z_SCORE_95 * judgeStd },
    ppi: { mean: trueMean, halfWidth: Z_SCORE_95 * ppiStd },
    effectiveSampleSize: effectiveSampleSize,
    powerMultiplier: effectiveSampleSize / humanSize,
  };
  return result;
}
