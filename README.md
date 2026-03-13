

<p align="center">
  <a href="https://github.com/EmertonData/glide/actions/workflows/ci.yml"><img src="https://github.com/EmertonData/glide/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/EmertonData/glide"><img src="https://codecov.io/gh/EmertonData/glide/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scientific-python.org/specs/spec-0000/"><img src="https://img.shields.io/badge/SPEC-0-green?labelColor=grey" alt="SPEC 0"></a>
  <img src="https://img.shields.io/badge/docs-passing-brightgreen" alt="Docs">
  <img src="https://img.shields.io/badge/python-3.12+-blue" alt="Python 3.12+">
  <a href="https://pypi.org/project/glide-py/"><img src="https://img.shields.io/pypi/v/glide-py"></a>
  <a href="https://github.com/EmertonData/glide/releases"><img src="https://img.shields.io/github/v/release/EmertonData/glide"></a>
  <a href="https://github.com/EmertonData/glide/commits/master"><img src="https://img.shields.io/github/commits-since/EmertonData/glide/latest"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/assets/logo-glide.png" alt="GLIDE Logo" width="80%">
</p>

# GLIDE 
### Generative Label Inference & Debiasing Engine

## 🧭 What is GLIDE?

GLIDE is a Python library for **rigorous evaluation of GenAI systems** using hybrid human/proxy annotations.

## 🤔 Why GLIDE?

- 🤖 **GenAI applications are everywhere — and imperfect.** Deployed systems make mistakes, and measuring how often matters.
- ⚖️ **LLM-as-judge is biased.** Proxy evaluators (models, heuristics) are cheap but systematically over- or under-estimate true performance.
- 🧑 **Rigorous evaluation requires a human in the loop.** Ground-truth labels from humans are expensive, so only a small subset is feasible.
- 📐 **GLIDE bridges the gap.** It combines a small set of human annotations with a large set of proxy predictions to produce statistically valid metrics — correcting proxy bias without requiring full human labeling.

## ⚡ Quick Start

```bash
pip install glide-py
```

## 📚 Documentation

[Documentation](https://glide-py.readthedocs.io/en/latest/)

## 🤝 Contributing

Contributions are welcome! Feel free to open an [issue](https://github.com/EmertonData/glide/issues) to report a bug or suggest a feature, or submit a [pull request](https://github.com/EmertonData/glide/pulls) with your changes.

## 🔢 Versioning

This project follows [Semantic Versioning (SemVer)](https://semver.org/): `MAJOR.MINOR.PATCH`.

## 📦 Dependency Support

This project follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) for dependency support windows.

## 📄 License & Citation

This project is licensed under the [Apache 2.0 License](LICENSE).

If you use Glide in your research, please cite:

```bibtex
@software{glide,
  title  = {GLIDE: Generative Label Inference \& Debiasing Engine},
  year   = {2026},
  url    = {https://github.com/EmertonData/glide},
}
```

## 🏛️ Affiliation

Developed at [Emerton Data](https://www.emerton-data.com/).

<img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/assets/logo-ed.jpg" alt="Emerton Data" width="250">
