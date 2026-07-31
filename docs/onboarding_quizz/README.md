# Onboarding Quizzes

Validation quizzes for the [onboarding checklist](../doc-onboarding-checklist.md), used to check that a
checkpoint's chapters were properly understood. Each quiz is multiple-choice with a 80%
pass threshold and gives immediate per-question feedback (correct answer plus explanation) on submit.

## Setup

To load the quizz, serve the folder over HTTP via:

```bash
cd docs/onboarding_quizz
python3 -m http.server 8000
```

Then open `http://localhost:8000/quizz1.html` (or `quizz2.html`).

## Taking a quiz

Answer every question and click **Submit**. Each question shows whether you got it right, with the correct
answer and explanation if not. Your total score and pass/fail status appear at the bottom. There's no
persistence: reload the page to retake a quiz.
