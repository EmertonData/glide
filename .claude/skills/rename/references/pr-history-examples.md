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

Lesson: a rename only touches direct references to the symbol, not variables a human (or a previous LLM pass) independently named after it. Grep for the old name as a substring of variable names too, not only as an exact symbol reference.

## 2. Plot color constants and their own inline comments

```python
# before
C_PPI = "#27AE60"  # PPI monitor running mean/bound        — green
C_ALARM_PPI = "#8E44AD"  # PPI monitor alarm               — purple

# after
C_PPRM = "#27AE60"  # PPRM running mean/bound              — green
C_ALARM_PPRM = "#8E44AD"  # PPRM alarm                     — purple
```

Lesson: a comment describing a renamed symbol needs the same rename applied to its text, and any manual alignment tied to the old text's length (here, the trailing `—` column) needs to be re-padded once the length changes.

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

Lesson: a display string used as a dict key must be renamed at every definition and lookup site in lockstep (here, later lookups like `raw_stats[correlation]["Cluster PPI++"]`), or the mismatch surfaces at runtime as a `KeyError` instead of at diff time. `make test-notebooks` catches this class of stray even when a grep pass scoped to the class name alone misses the separate display-string variant.

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

Lesson: a notebook's title is prose in its first cell, not code, so a code-level rename never touches it automatically. It requires its own explicit edit, and a diff review that starts from the code cells downward will skip over it.

## 6. Stale references in CONTRIBUTING.md's architecture tree

```diff
 ├── monitors/               # Public API — drift monitors over batched data
-│   ├── ppi.py
+│   ├── empirical_ppi.py
 │   ├── ...
```

Lesson: an ASCII directory tree inside a markdown file is prose-shaped but encodes real file paths; a grep pass scoped to code references alone will miss it. `CONTRIBUTING.md` and `README.md` need their own dedicated grep pass for this reason.
