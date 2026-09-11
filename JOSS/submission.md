# JOSS submission form — proposed content

Form: <https://joss.theoj.org/papers/new>. Everything below is copy-paste ready; the two bracketed items must be settled before submitting.

## Form fields

**Title**

```
GLIDE: A Python library for prediction-powered evaluation of GenAI systems
```

**Software's Git repository URL**

```
https://github.com/EmertonData/glide
```

**Name of git branch containing the paper, not the path**

Leave blank. `JOSS/paper.md` will live on the default branch (`main`), and JOSS locates `paper.md` anywhere in the repository, so the path itself is never entered on the form.

**Software version**

```
v0.11.0
```

**Type of submission**

`New submission` (the three options are New submission / Resubmission / Major new version).

**Main subject of the paper**

Start typing `AI safety` and select:

```
AI safety, alignment, and evaluation
```

This routes the paper to the *Data Science, Artificial Intelligence, and Machine Learning* track. If the editors prefer a more statistical routing, `Statistical software and libraries` in the same track is the natural alternative.

## Message to editors

```
GLIDE is a Python library for prediction-powered inference specialized to mean
estimation: it combines a small set of human annotations with a large pool of
LLM-as-judge (proxy) labels to produce debiased estimates of evaluation metrics
with valid confidence intervals, and extends the same primitives to
anytime-valid monitoring of a metric tracked over successive batches of
production data.

Prior and planned publication. A companion methods paper describing the
statistical framework behind the library was presented at the ICML 2026
workshop on statistical frameworks for uncertainty in agentic systems, which is
non-archival, and is available as a preprint (arXiv:2605.31278, linked from the
repository README badges). No portion of the submitted work is under review at,
or planned for submission to, a peer-reviewed venue. The JOSS paper is distinct
in content and purpose: it describes the library itself, its architecture and
design trade-offs, its data contract, its testing and validation
infrastructure, and its documentation, whereas the preprint presents the
statistical methodology and its empirical validation, which the JOSS paper only
summarizes.

Development history. The repository has been public since its first commit on
5 March 2026, and development has been entirely in the open since: eleven
releases, over 340 merged pull requests, 70 public issues, seven contributors, an
issue-first contribution guide with templates, a code of conduct, automated
dependency updates, and continuous integration enforcing linting, type
checking, 100% unit-test coverage, doctests, functional statistical tests and
execution of every documentation notebook.

Generative AI. As disclosed in the paper's AI usage disclosure section, Claude
Code was used as a coding assistant for implementation, tests, documentation
and manuscript drafting, while problem framing, method selection, architectural
decisions, statistical validation and code review were performed by the human
authors, who verified and take responsibility for all output.

Dissemination. GLIDE is the subject of a tutorial at PyData Amsterdam on
12 September 2026 and of a talk at Compute! Paris on 25 November 2026.

Conflicts of interest. All authors are employed by Emerton Data, which funded
the development of GLIDE and releases it under the Apache-2.0 license; the
company sells no product that depends on the library. We are aware of no other
conflict of interest. As potential reviewer conflicts, we note that the library
implements methods whose authors we cite extensively, none of whom has been
involved in this work.
```

## Pre-submission checkboxes

- **"I certify that I am submitting software for which I am a primary author"** — tick (Grégoire Martinon, corresponding author).
- **"I have verified that my paper compiles using one of these tools"** — tick only after the `Draft paper PDF` workflow has produced a clean `paper.pdf` artifact (see below).
- **"I confirm that I read and will adhere to the JOSS code of conduct"** — tick.

## Before pressing submit

1. **[to settle] ORCIDs.** `JOSS/paper.md` carries `0000-0000-0000-0000` placeholders for both authors. JOSS strongly prefers a real ORCID for every author and validates the checksum, so these must be replaced.
2. **[to settle] Submission date.** The `date:` field in `JOSS/paper.md` reads `8 September 2026`. Set it to the actual submission date, which must be on or after **5 September 2026** to satisfy JOSS's requirement of six months of public repository history.
3. Release `v0.11.0` and confirm the tag exists, since the version entered on the form is the one reviewers will check out.
4. Run the `Draft paper PDF` workflow and confirm the PDF renders, in particular the figure and the reference list.
5. Confirm that the co-authors listed in `paper.md` agree to being listed, and that the contributors credited in the Acknowledgements are happy with that credit rather than authorship.

## Compiling the paper: the `Draft paper PDF` GitHub Action

JOSS asks submitters to verify that `paper.md` compiles before submitting. This repository now does that in CI, in `.github/workflows/draft_paper.yml`.

The workflow calls `openjournals/openjournals-draft-action@v1.0`, the official Open Journals action. It is a Docker action that runs the `openjournals/inara` image, the same Pandoc-based toolchain JOSS uses in production to typeset accepted papers, so a PDF that builds here builds on the JOSS side too. It takes two inputs: `journal` (`joss` or `jose`) and `paper-path`, the path to the Markdown source relative to the repository root. Inara writes its output next to the source file, so the artifact is uploaded from `JOSS/paper.pdf`.

Behavior worth knowing:

- **Triggers.** Pushes to `main` and pull requests against `main` that touch `JOSS/**` or the workflow file, plus `workflow_dispatch` so the PDF can be rebuilt on demand from the Actions tab. Scoping by path keeps unrelated pushes from rebuilding the same PDF.
- **Output.** The PDF is published as an Actions artifact named `joss-paper`, downloadable from the run summary page. Nothing is committed back to the repository, which is why the job keeps `permissions: contents: read`.
- **Version pinning.** The action is pinned to `v1.0`, matching the repository's convention of pinning every action. Note that `v1.0` internally references `openjournals/inara:latest`, so the container itself floats; that is deliberate on Open Journals' side, and it means the build always reflects the current JOSS toolchain rather than a frozen one.
- **Failure modes to expect.** Unresolvable citation keys, a malformed YAML header, and image paths that Inara cannot resolve. The figure is therefore stored inside `JOSS/` and referenced as `glide-workflow.png`, without a `../` prefix.
- **Local equivalent.** `make paper` runs the same container against the same file and drops `JOSS/paper.pdf` in the working tree, which is faster than pushing when iterating on the text. It requires Docker.
