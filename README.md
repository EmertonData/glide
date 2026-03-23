

<p align="center">
  <a href="https://github.com/EmertonData/glide/actions/workflows/code_quality.yml"><img src="https://github.com/EmertonData/glide/actions/workflows/code_quality.yml/badge.svg" alt="Code quality"></a>
  <a href="https://codecov.io/gh/EmertonData/glide"><img src="https://codecov.io/gh/EmertonData/glide/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scientific-python.org/specs/spec-0000/"><img src="https://img.shields.io/badge/SPEC-0-green?labelColor=grey" alt="SPEC 0"></a>
  <a href="https://glide-py.readthedocs.io/en/latest/"><img src="https://app.readthedocs.org/projects/glide-py/badge/?version=stable" alt="Docs"></a>
  <a href="https://pypi.org/project/glide-py/"><img src="https://img.shields.io/pypi/pyversions/glide-py" alt="Python versions"></a>
  <a href="https://pypi.org/project/glide-py/"><img src="https://img.shields.io/pypi/v/glide-py" alt="PyPI"></a>
  <a href="https://github.com/EmertonData/glide/releases"><img src="https://img.shields.io/github/v/release/EmertonData/glide" alt="Release"></a>
  <a href="https://github.com/EmertonData/glide/commits/master"><img src="https://img.shields.io/github/commits-since/EmertonData/glide/latest" alt="Commits"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/assets/logo-glide.png" alt="GLIDE Logo" width="80%">
</p>

# GLIDE 
### Generated Label Inference & Debiasing Engine

## 🧭 What is GLIDE?

GLIDE is a Python library for **rigorous evaluation of GenAI systems** using hybrid human/proxy annotations.

GLIDE implements methods from the field of **semi-supervised Inference** — the science of system evaluation that combines a small labeled dataset with a large unlabeled (or proxy-labeled) dataset to produce valid, debiased estimates. See the [implemented papers](#implemented-papers) below.

## 🤔 Why GLIDE?

- 🤖 **GenAI applications are everywhere — and imperfect.** Deployed systems make mistakes, and measuring how often matters.
- ⚖️ **LLM-as-judge is biased.** Proxy evaluators (models, heuristics) are cheap but systematically over- or under-estimate true performance.
- 🧑 **Rigorous evaluation requires a human in the loop.** Ground-truth labels from humans are expensive, so only a small subset is feasible.
- 📐 **GLIDE bridges the gap.** It combines a small set of human annotations with a large set of proxy predictions to produce statistically valid metrics — correcting proxy bias without requiring full human labeling.

## ⚡ Quick Start
Install the package with your favorite package manager :

```bash
uv add glide-py
```
or
```bash
pip install glide-py
```
And look at our practical [quickstart](https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/docs/getting-started/quickstart.ipynb).

## 📚 Documentation

Explore the full [documentation](https://glide-py.readthedocs.io/en/latest/) — from practical tutorials and a user guide to scientific deep dives into the methods behind GLIDE.

## 🤝 Contributing

Contributions are welcome! Please read the [contributing guide](https://github.com/EmertonData/glide/blob/main/CONTRIBUTING.md) for setup instructions, an architectural overview, and the checklist to follow before opening a pull request. Feel free to open an [issue](https://github.com/EmertonData/glide/issues) to report a bug or suggest a feature.

## 🔢 Versioning

This project follows [Semantic Versioning (SemVer)](https://semver.org/): `MAJOR.MINOR.PATCH`.

## 📦 Dependency Support

This project follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) for dependency support windows.

## 📄 License & Citation

This project is licensed under the [Apache 2.0 License](https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/LICENSE).
If you use Glide in your research, please cite:

```bibtex
@software{glide,
  title  = {GLIDE: Generated Label Inference \& Debiasing Engine},
  year   = {2026},
  url    = {https://github.com/EmertonData/glide},
}
```

## 📰 Implemented Papers <a name="implemented-papers"></a>

| Year | Title | Venue | Original Implementation | GLIDE class |
|------|-------|-------|------|----------------|
| 2023 | [Prediction-powered inference](https://www.science.org/doi/10.1126/science.adi6000) |Science|[Link](https://github.com/aangelopoulos/ppi_py/)| estimators.PPIMeanEstimator (with `power_tuning=False`) |
| 2023 | [PPI++: Efficient Prediction-Powered Inference](https://arxiv.org/abs/2311.01453) |Preprint|[Link](https://github.com/aangelopoulos/ppi_py/tree/ppi++)| estimators.PPIMeanEstimator |
| 2024 | [Active Statistical Inference](https://dl.acm.org/doi/10.5555/3692070.3694680) |ICML'24|[Link](https://github.com/tijana-zrnic/active-inference)| estimators.ASIMeanEstimator |
| 2025 | [Can Unconfident LLM Annotations Be Used for Confident Conclusions?](https://aclanthology.org/2025.naacl-long.179/) |NAACL'25|[Link](https://github.com/kristinagligoric/confidence-driven-inference)| estimators.ASIMeanEstimator |





## 🏛️ Affiliation

Developed at [Emerton Data](https://www.emerton-data.com/).

<img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/assets/logo-ed.jpg" alt="Emerton Data" width="250">
