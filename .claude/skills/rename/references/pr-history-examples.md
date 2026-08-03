# Worked examples from GLIDE's rename history

Concrete before/after snippets pulled from real GLIDE renames and their follow-up "caught stray X" commits. Use these to sanity-check that you're catching the same categories of stray reference, not just the literal symbol name.

## 1. Notebook variable names outlive the class rename

A monitor class had already been renamed to `EmpiricalPPRM`/`AsymptoticPPRM` in an earlier commit, but a local variable in the tutorial notebooks was still named after the old acronym:

```python
# before (stray, caught in a later cleanup pass)
ppi_result_no_drift = AsymptoticPPRM().detect(...)
print(f"PPI monitor drift detected: {ppi_result_no_drift.drift_detected}")

# after
pprm_result_no_drift = AsymptoticPPRM().detect(...)
print(f"PPRM drift detected: {pprm_result_no_drift.drift_detected}")
```

Lesson: renaming the class does nothing to the variable a human (or a previous LLM pass) happened to name after the *old* class. Grep for the old name as a variable-name substring too, not just as a class reference.

## 2. Plot color constants and their own inline comments

```python
# before
C_PPI = "#27AE60"  # PPI monitor running mean/bound        — green
C_ALARM_PPI = "#8E44AD"  # PPI monitor alarm               — purple

# after
C_PPRM = "#27AE60"  # PPRM running mean/bound              — green
C_ALARM_PPRM = "#8E44AD"  # PPRM alarm                     — purple
```

Lesson: the comment text itself needs the rename too, and comment alignment (the trailing `—` column) often needs re-padding once the label text length changes.

## 3. Dict keys used as legend labels

```python
# before
colors = {"True only": "steelblue", "Cluster PPI++": "darkorange", "Proxy only": "red"}
...
ax.plot(correlations, ess_mean, marker="o", color="darkorange", label="Cluster PPI++ ESS (mean)")

# after
colors = {"True only": "steelblue", "Clustered PPI++": "darkorange", "Proxy only": "red"}
...
ax.plot(correlations, ess_mean, marker="o", color="darkorange", label="Clustered PPI++ ESS (mean)")
```

These dict keys are also used later as dictionary lookups (`raw_stats[correlation]["Cluster PPI++"]`) — every lookup site has to change in lockstep or the notebook throws a `KeyError` at run time. This is a case where `make test-notebooks` catches what grep might miss if you only search for the class name and not the display string.

## 4. Words derived from the same root, not the literal old identifier

The rename was `ClusterPPIMeanEstimator` → `ClusteredPPIMeanEstimator`, but plain prose using the adjective "cluster" (not the class name) also needed the same word change:

```markdown
<!-- before -->
- **True only** | `y_true` (annotated clusters) | Cluster classical estimator, the gold standard for validity |

<!-- after -->
- **True only** | `y_true` (annotated clusters) | Clustered classical estimator, the gold standard for validity |
```

Lesson: decide the "root word" for the whole rename (`cluster` → `clustered`) and audit prose using that root, not only occurrences of the exact class/function token.

## 5. Notebook titles lag behind the code rename

```markdown
<!-- before -->
# Scientific Validity of Asymptotic PPI Mean Monitoring

<!-- after -->
# Scientific Validity of Asymptotic PPRM
```

This needed a dedicated follow-up commit after the main rename — the H1 title of the notebook doesn't get touched by a code-level rename and is easy to forget because it's the very first cell, often skipped when skimming a diff top-down.

## 6. Stale references in CONTRIBUTING.md's architecture tree

```diff
 ├── monitors/               # Public API — drift monitors over batched data
-│   ├── ppi.py
+│   ├── empirical_ppi.py
 │   ├── ...
```

This is prose-shaped (an ASCII tree inside a markdown file), so it's easy to miss when searching only for code references. `CONTRIBUTING.md` and `README.md` are always worth a dedicated grep pass.

## 7. Historical CHANGELOG entries corrected retroactively

`Cluster*` was renamed to `Clustered*` shortly after being introduced. The already-released `## [0.7.0]` entry (not just the new unreleased entry) named the old symbols, so it was corrected too:

```diff
 ## [0.7.0] – 2026-06-12

 ### ✨ Added
-- Cluster-level inference support: `ClusterPPIMeanEstimator`, `ClusterClassicalMeanEstimator`, `UniformClusterSampler`, ...
+- Cluster-level inference support: `ClusteredPPIMeanEstimator`, `ClusteredClassicalMeanEstimator`, `UniformClusteredSampler`, ...
```

This is unusual (changelog entries normally describe history as it happened) and was only done because the symbol hadn't been out in a release for long. Ask the user before doing this on a rename of something that's been stable for a while.
