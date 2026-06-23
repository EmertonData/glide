

<p align="center">
  <a href="https://github.com/EmertonData/glide/actions/workflows/code_quality.yml"><img src="https://github.com/EmertonData/glide/actions/workflows/code_quality.yml/badge.svg" alt="Code quality"></a>
  <a href="https://codecov.io/gh/EmertonData/glide"><img src="https://codecov.io/gh/EmertonData/glide/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://glide-py.readthedocs.io/en/latest/"><img src="https://app.readthedocs.org/projects/glide-py/badge/?version=stable" alt="Docs"></a>
  <a href="https://scientific-python.org/specs/spec-0000/"><img src="https://img.shields.io/badge/SPEC-0-green?labelColor=grey" alt="SPEC 0"></a>
  <a href="https://pypi.org/project/glide-py/"><img src="https://img.shields.io/pypi/pyversions/glide-py" alt="Python versions"></a>
  <a href="https://pypi.org/project/glide-py/"><img src="https://img.shields.io/pypi/v/glide-py" alt="PyPI"></a>
  <a href="https://github.com/EmertonData/glide/releases"><img src="https://img.shields.io/github/v/release/EmertonData/glide" alt="Release"></a>
  <a href="https://github.com/EmertonData/glide/commits/master"><img src="https://img.shields.io/github/commits-since/EmertonData/glide/latest" alt="Commits"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://www.linkedin.com/build-relation/newsletter-follow?entityUrn=7453423077557952512"><img src="https://img.shields.io/badge/Follow-LinkedIn-0A66C2" alt="LinkedIn"></a>
  <a href="https://arxiv.org/abs/2605.31278"><img src="https://img.shields.io/badge/arXiv-2605.31278-b31b1b" alt="arXiv"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/docs/assets/logo-glide-white-bg.png" alt="GLIDE Logo" width="80%">
</p>

# GLIDE 
### Generated Label Inference & Debiasing Engine

## 🧭 What is GLIDE?

GLIDE is a Python library for **rigorous evaluation of GenAI systems** using hybrid human/proxy annotations.

GLIDE implements methods from the field of **prediction-powered inference** — the science of system evaluation that combines a small set of labeled data with a large set of proxy-labeled data to produce valid, debiased estimates. See the [implemented algorithms](#implemented-algorithms) below.

<p align="center">
  <img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/docs/assets/schema-PPI-readme.png" alt="Prediction-powered inference schema" width="60%">
</p>

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
And look at our practical [quickstart](https://glide-py.readthedocs.io/en/latest/getting-started/quickstart/).

## 📚 Documentation

Explore the full [documentation](https://glide-py.readthedocs.io/en/latest/) — from practical tutorials and user guides to scientific deep dives into the methods behind GLIDE.

## 🤝 Contributing

Contributions are welcome! Please read the [contributing guide](https://github.com/EmertonData/glide/blob/main/CONTRIBUTING.md) for setup instructions, an architectural overview, and the checklist to follow before opening a pull request. Feel free to open an [issue](https://github.com/EmertonData/glide/issues) to report a bug or suggest a feature.

## 🔢 Versioning

This project follows [Semantic Versioning (SemVer)](https://semver.org/): `MAJOR.MINOR.PATCH`.

## 📦 Dependency Support

This project follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) for dependency support windows.

## 📄 License & Citation

This project is licensed under the [Apache 2.0 License](https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/LICENSE).

If you use GLIDE in your work, please cite us using the "Cite this repository" button on the GitHub repository page.

## 📚 Implemented Algorithms <a name="implemented-algorithms"></a>

| Name | Class | Reference Paper(s) | Original Implementation |
|------|-------|-----------------|------------------------|
| Prediction-Powered Inference | `estimators.PPIMeanEstimator` (with `power_tuning=False`) | [[1]](#ref-1) | [Link](https://github.com/aangelopoulos/ppi_py/) |
| PPI++ | `estimators.PPIMeanEstimator` | [[2]](#ref-2) | [Link](https://github.com/aangelopoulos/ppi_py/tree/ppi++) |
| Stratified Prediction-Powered Inference | `estimators.StratifiedPPIMeanEstimator` | [[3]](#ref-3) | — |
| Clustered Prediction-Powered Inference | `estimators.ClusteredPPIMeanEstimator` | — | [Link](https://github.com/davidbroska/ppi_py) |
| Multi-Proxy Prediction-Powered Inference | `estimators.MultiPPIMeanEstimator` | [[8]](#ref-8) | [Link](https://github.com/jw-shan/sada) |
| Stratified Sampling | `samplers.StratifiedSampler` | [[4]](#ref-4) | [Link](https://github.com/amazon-science/ssepy) |
| Active Statistical Inference | `estimators.ASIMeanEstimator` | [[5]](#ref-5), [[6]](#ref-6) | [Link](https://github.com/tijana-zrnic/active-inference) |
| Active Sampling | `samplers.ActiveSampler` | [[5]](#ref-5), [[6]](#ref-6) | [Link](https://github.com/kristinagligoric/confidence-driven-inference) |
| Predict-Then-Debias | `estimators.PTDMeanEstimator` | [[7]](#ref-7) | [Link](https://github.com/DanKluger/PTDBoot) |
| Stratified Predict-Then-Debias | `estimators.StratifiedPTDMeanEstimator` | [[7]](#ref-7) | [Link](https://github.com/DanKluger/PTDBoot) |
| Clustered Predict-Then-Debias | `estimators.ClusteredPTDMeanEstimator` | [[7]](#ref-7) | [Link](https://github.com/DanKluger/PTDBoot) |
| IPW Predict-Then-Debias | `estimators.IPWPTDMeanEstimator` | [[7]](#ref-7) | [Link](https://github.com/DanKluger/PTDBoot) |

### 📖 References

<a id="ref-1"></a>[1] <a href="https://www.science.org/doi/10.1126/science.adi6000">Angelopoulos, Anastasios N., Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. "Prediction-powered inference." Science 382, no. 6671 (2023): 669-674.</a>

<a id="ref-2"></a>[2] <a href="https://arxiv.org/abs/2311.01453">Angelopoulos, Anastasios N., John C. Duchi, and Tijana Zrnic. "PPI++: Efficient prediction-powered inference." arXiv preprint arXiv:2311.01453 (2023).</a>

<a id="ref-3"></a>[3] <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract-Conference.html">Fisch, Adam, Joshua Maynez, R. Alex Hofer, Bhuwan Dhingra, Amir Globerson, and William W. Cohen. "Stratified prediction-powered inference for effective hybrid evaluation of language models." Advances in Neural Information Processing Systems 37 (2024): 111489-111514.</a>

<a id="ref-4"></a>[4] <a href="https://link.springer.com/chapter/10.1007/978-3-031-73223-2_9">Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework for efficient model evaluation through stratification, sampling, and estimation." In European Conference on Computer Vision, pp. 140-158. Cham: Springer Nature Switzerland, 2024.</a>

<a id="ref-5"></a>[5] <a href="https://dl.acm.org/doi/10.5555/3692070.3694680">Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." In Proceedings of the 41st International Conference on Machine Learning, pp. 62993-63010. 2024.</a>

<a id="ref-6"></a>[6] <a href="https://aclanthology.org/2025.naacl-long.179/">Gligorić, Kristina, Tijana Zrnic, Cinoo Lee, Emmanuel Candes, and Dan Jurafsky. "Can unconfident LLM annotations be used for confident conclusions?" In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3514-3533. 2025.</a>

<a id="ref-7"></a>[7] <a href="https://arxiv.org/abs/2501.18577">Kluger, Dan M., Kerri Lu, Tijana Zrnic, Sherrie Wang, and Stephen Bates. "Prediction-powered inference with imputed covariates and nonuniform sampling." arXiv preprint arXiv:2501.18577 (2025).</a>

<a id="ref-8"></a>[8] <a id="ref-8-link" href="https://arxiv.org/abs/2509.21707">Shan, Jiawei, Zhifeng Chen, Yiming Dong, Yazhen Wang, and Jiwei Zhao. "SADA: Safe and Adaptive Aggregation of Multiple Black-Box Predictions in Semi-Supervised Learning." arXiv preprint arXiv:2509.21707 (2025).</a>.

## 📬 Stay Updated

Follow our [LinkedIn newsletter](https://www.linkedin.com/build-relation/newsletter-follow?entityUrn=7453423077557952512) for updates on GLIDE and GenAI evaluation.

## 🏛️ Affiliation

Developed at [Emerton Data](https://www.emerton-data.com/).

<img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/docs/assets/logo-ed.jpg" alt="Emerton Data" width="20%">
