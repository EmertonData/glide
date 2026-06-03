# Feature ticket template

```markdown
# [Feature title: algorithm name and brief purpose]

## TODO

- [ ] <!-- placeholder for refinement actions: update, split, convert to issue, etc. -->

## Background

[2–3 tight sentences per paragraph, 2 paragraphs max. Answer: what problem does this solve, what is the core mathematical idea, and why it matters for GLIDE. Write for a developer with no statistics background — define any term that is not common Python knowledge. One formula at most, introduced with "In math notation this is written as...". Never summarise the full paper.]

## Design choices

[1–2 paragraphs, strictly synthetic. Cover only: where in the codebase this lives and why, how it relates to existing classes (inherits from X, mirrors Y), and any key interface decisions. One sentence of reasoning per choice is enough. Skip anything the code already makes obvious.]

## Implementation

[For each class or function in scope, write a block like the one below. The code here is a working blueprint: complete signatures, complete NumPy docstrings, concrete tentative implementation with inline comments that walk through the algorithm step by step. A developer should be able to run this without reading the paper.]

### ClassName

**Location:** `glide/<module>/<file>.py`

**Purpose:** [One sentence.]

```python
from typing import Optional, List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray


class ClassName:
    """One-line summary.

    [Longer description: what it computes, when to use it.]

    Parameters
    ----------
    param_name : type
        Description.

    Examples
    --------
    >>> from glide.estimators import ClassName
    >>> estimator = ClassName(param=value)
    >>> result = estimator.estimate(labeled, proxy)
    1.5
    """

    def __init__(self, param: Type) -> None:
        self.param = param

    def estimate(self, labeled: NDArray, proxy: NDArray) -> float:
        """One-line summary.

        Parameters
        ----------
        labeled : NDArray
            Description.
        proxy : NDArray
            Description.

        Returns
        -------
        float
            Description.

        Examples
        --------
        >>> from glide.estimators import ClassName
        >>> estimator = ClassName(param=value)
        >>> estimator.estimate(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        1.5
        """
        # [Explain this step in plain language, e.g. "Compute the correction term that removes proxy bias"]
        correction = np.mean(labeled - proxy[:len(labeled)])
        # [Explain the final estimate]
        result = np.mean(proxy) + correction
        return result
```

[Repeat for each class or standalone function in scope.]

## Functional tests

[Include this section to specify mathematical properties the new object must satisfy beyond basic code correctness. These can be equivalences, ordering guarantees, monotonicity properties, or any other theoretically grounded expectation. Omit entirely if no such property applies.

For each property, state it plainly and show the corresponding test structure.]

**[Name of the property, e.g. "Single-stratum stratified PPI equals plain PPI"]**

[One sentence stating the property: "When all observations belong to a single stratum, StratifiedPPIMeanEstimator must return the same value as PPIMeanEstimator."]

```python
def test_single_stratum_equals_ppi():
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.1, 1.9, 2.1, 0.9])
    groups = np.array(["a", "a", "a", "a"])

    result_stratified = StratifiedPPIMeanEstimator(...).estimate(y_true, y_proxy, groups)
    result_ppi = PPIMeanEstimator(...).estimate(y_true, y_proxy)

    np.testing.assert_allclose(result_stratified, result_ppi, atol=1e-10)
```

**[Name of an ordering property, e.g. "Stratified PPI confidence interval is no wider than plain PPI"]**

[One sentence: "For the same data, StratifiedPPIMeanEstimator must produce a confidence interval at least as tight as PPIMeanEstimator."]

```python
def test_stratified_ppi_tighter_than_ppi():
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.1, 1.9, 2.1, 0.9])
    groups = np.array(["a", "a", "b", "b"])

    ci_stratified = StratifiedPPIMeanEstimator(...).estimate(y_true, y_proxy, groups)
    ci_ppi = PPIMeanEstimator(...).estimate(y_true, y_proxy)

    assert ci_stratified.width <= ci_ppi.width
```

## Corner cases

- [Specific scenario and expected behaviour — raise, clamp, warn, or return a defined value. Be concrete about inputs and outputs.]
- [Another scenario]
- [...]

## Acceptance criteria

- [ ] [Specific deliverable tied to this ticket — name the class, method, or behaviour. E.g.: "ClusterPTDMeanEstimator.estimate returns a CI that contains the true mean in >95% of simulations", or "_preprocess raises ValueError when fewer than 2 clusters are selected".]
- [ ] [Another specific deliverable — numerical validation, equivalence property, or observable behaviour.]
- [ ] [...]

Do not include items already in the PR template: `make lint`, `make type-check`, `make coverage`, `make doc`, and `CHANGELOG.md` are required on every PR and belong there, not here.
```
