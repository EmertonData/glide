export const Z_SCORE_95 = 1.959963984540054;

function computeCovarianceBounds(trueMean, proxyMean) {
  const lowerCovariance = Math.max(0, trueMean + proxyMean - 1) - trueMean * proxyMean;
  const upperCovariance = Math.min(trueMean, proxyMean) - trueMean * proxyMean;
  return { lowerCovariance, upperCovariance };
}

function clampCovariance(rawCovariance, trueMean, proxyMean) {
  const { lowerCovariance, upperCovariance } = computeCovarianceBounds(trueMean, proxyMean);
  const clamped = Math.min(Math.max(rawCovariance, lowerCovariance), upperCovariance);
  return clamped;
}

/**
 * Feasible human/judge correlation range for the given Bernoulli accuracies.
 *
 * @param {number} trueMean   Human accuracy p in (0, 1).
 * @param {number} proxyMean  Judge accuracy q in (0, 1).
 * @returns {{minCorrelation: number, maxCorrelation: number}} The Fréchet bounds on the correlation.
 */
export function getCorrelationBounds(trueMean, proxyMean) {
  const trueVariance = trueMean * (1 - trueMean);
  const proxyVariance = proxyMean * (1 - proxyMean);
  const { lowerCovariance, upperCovariance } = computeCovarianceBounds(trueMean, proxyMean);
  const standardDeviationProduct = Math.sqrt(trueVariance * proxyVariance);
  const bounds = {
    minCorrelation: lowerCovariance / standardDeviationProduct,
    maxCorrelation: upperCovariance / standardDeviationProduct,
  };
  return bounds;
}

/**
 * Simulate the three estimators (human-only, judge-only, PPI) from population parameters.
 *
 * @param {Object} params
 * @param {number} params.totalSize      Total proxy-labeled dataset size N.
 * @param {number} params.humanSize      Human-labeled subset size n, 0 < n < N.
 * @param {number} params.trueMean       Human accuracy p in (0, 1).
 * @param {number} params.proxyMean      Judge accuracy q in (0, 1).
 * @param {number} params.correlation    Human/judge correlation rho in [0, 1].
 * @returns {{humanOnly: {mean: number, halfWidth: number}, judgeOnly: {mean: number, halfWidth: number}, ppi: {mean: number, halfWidth: number}, effectiveSampleSize: number, powerMultiplier: number}}
 */
export function simulate({ totalSize, humanSize, trueMean, proxyMean, correlation }) {
  if (!(humanSize > 0 && humanSize < totalSize)) {
    throw new Error(`'humanSize' must satisfy 0 < humanSize < totalSize; got ${humanSize}.`);
  }
  const validations = [
    ["trueMean", trueMean, (value) => value > 0 && value < 1, "(0, 1)"],
    ["proxyMean", proxyMean, (value) => value > 0 && value < 1, "(0, 1)"],
    ["correlation", correlation, (value) => value >= 0 && value <= 1, "[0, 1]"],
  ];
  for (const [name, value, isValid, rangeDescription] of validations) {
    if (!isValid(value)) {
      throw new Error(`'${name}' must be in ${rangeDescription}; got ${value}.`);
    }
  }

  const numberUnlabeled = totalSize - humanSize;
  const trueVariance = trueMean * (1 - trueMean);
  const proxyVariance = proxyMean * (1 - proxyMean);
  const rawCovariance = correlation * Math.sqrt(trueVariance * proxyVariance);
  const labeledCovariance = clampCovariance(rawCovariance, trueMean, proxyMean);

  const factor = 1 + humanSize / numberUnlabeled;
  const tuningParameter = labeledCovariance / (factor * proxyVariance);

  const residualVariance = trueVariance - 2 * tuningParameter * labeledCovariance + tuningParameter ** 2 * proxyVariance;
  const ppiVariance = residualVariance / humanSize + (tuningParameter ** 2 * proxyVariance) / numberUnlabeled;
  const ppiStd = Math.sqrt(ppiVariance);

  const humanStd = Math.sqrt(trueVariance / humanSize);
  const judgeStd = Math.sqrt(proxyVariance / totalSize);

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
