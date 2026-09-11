# GenAI for Finance — X-HEC Training Plan

One-day training, four sessions: **Presentation → Practice → Presentation → Practice**, followed by a short closing block.
Narrative arc: *what is this technology → how do you build with it → how do you know it's any good → prove it yourself.*

---

## Session 1 — Presentation: Introduction & How to Build

1. **GenAI for Finance: Building and Evaluating Trustworthy AI Systems** — title slide; frames the day as two halves, build then evaluate.
2. **A new general-purpose technology** — the AI/electricity analogy: general-purpose technologies reshape every industry gradually, not overnight, and GenAI in finance is early in that curve.
3. **What is an LLM, really** — an LLM is fundamentally a next-word predictor trained on massive text corpora.
4. **Why next-word prediction is so powerful** — language encodes sentences, ideas, and executable actions, so predicting text becomes a general-purpose reasoning and acting engine.
5. **The vocabulary you need today** — token, prompt, chunk, embedding, context window, tool, agent: the building blocks used in every exercise this afternoon.
6. **Two things to know about where the field is headed** — tokens are getting cheaper and models are tackling increasingly complex, multi-step tasks.
7. **GenAI in finance: three entry points** — due diligence, market screening, and fraud investigation are where GenAI is already changing finance workflows.
8. **Today's thread: due diligence** — introduces the running exercise: build an agentic RAG that answers due-diligence questions over a real financial document.
9. **What happens at inference time** — generating a token is a single forward pass through a fixed, frozen model; nothing is learned live.
10. **How LLMs are trained (a pointer, not a detour)** — pretraining, fine-tuning, and alignment turn raw text into an assistant; for the full mechanics, see Sebastian Raschka's *Build a Large Language Model from Scratch* — not covered live today.
11. **Why retrieve at all: Retrieval-Augmented Generation** — RAG grounds the model's answers in your own documents instead of its frozen, sometimes outdated training data.
12. **Chunking: the decision that shapes everything downstream** — how you cut a document into pieces determines what can and cannot be retrieved later.
13. **Hybrid search: semantic meaning + exact keywords** — combining dense embeddings with keyword search (BM25) covers both "meaning" queries and exact-match queries like a specific revenue figure.
14. **From RAG to agents: giving the model a tool** — an agent is an LLM that can decide, on its own, to call a tool (like a search function) before answering.
15. **The ReAct loop** — reason, act, observe, repeat: the simplest and most common agentic pattern, and the one you'll build this afternoon.
16. **Walkthrough: what you're about to build** — a live tour of the exercise notebook's five building blocks (chunk, embed, search, ReAct loop, answer) before students start.

---

## Session 2 — Practice: Build Your Own Agentic RAG

Students complete a fill-in-the-blank notebook that implements `chunk_text`, `vectorize_text`, a hybrid BM25 + semantic search tool, and a single ReAct agent loop, then run their finished agent against due-diligence questions on the workshop's sample financial document.

---

## Session 3 — Presentation: How to Evaluate

1. **From Building to Trusting** — transition slide: how do you know your RAG is actually any good?
2. **The GenAI risk landscape** — hallucination, staleness, and inconsistency are the core risks that make GenAI outputs different from traditional software outputs.
3. **Performance vs. reliability** — a system can be accurate on average and still be unreliable case by case; evaluation must capture both.
4. **Evaluating a RAG response: the RAG triad** — faithfulness, answer relevance, and context relevance give a three-part diagnostic for where a RAG pipeline is failing.
5. **Faithfulness, front and center** — for due diligence, the question that matters most is whether the answer sticks to the retrieved evidence or invents beyond it; this is the metric the rest of the afternoon is built around.
6. **Who judges faithfulness: LLM-as-judge** — a strong model can grade faithfulness at scale, cheaply, but it carries its own biases.
7. **The catch: a judge's bias needs correcting, not ignoring** — treating judge scores as ground truth silently bakes the judge's own errors into every conclusion you draw from them.
8. **Combining cheap and expensive signal: Prediction-Powered Inference** — PPI lets you combine a few expensive, high-quality labels with many cheap LLM-judge labels into one statistically valid estimate.
9. **GLIDE: what it is, when to use it, how to use it** — GLIDE packages PPI-style estimators so you can plug in your own labeled and proxy-labeled faithfulness scores and get a debiased estimate with a confidence interval.
10. **Beyond correctness: evaluating productivity** — the ultimate question isn't just "is the answer right" but "did it make the analyst faster" — time-to-success as a broader perspective on what "good" means.
11. **Walkthrough: what you're about to do** — a preview of the afternoon: score a provided dataset's faithfulness with a judge, then debias that judge with GLIDE.

---

## Session 4 — Practice: Measure Faithfulness with GLIDE

Using a provided dataset (paragraph/claim pairs with faithfulness labels, independent of students' own Session 2 agent), students score faithfulness with an LLM judge and a small set of gold-labeled examples, then use GLIDE to combine both into a single debiased faithfulness estimate with a confidence interval — the statistical core of the day, and the one thing students should walk away remembering.

---

## Closing (30 min, end of day)

1. **★ HIGHLIGHT — From today's $200 to the planet's footprint** — start from the concrete number students just watched (today's own API spend, and the energy behind it), then zoom out to what AI's global energy footprint actually looks like; the small, honest local number makes the global scale legible instead of abstract.
2. **Will this replace your job?** — the evidence points more toward task automation and augmentation than wholesale job replacement, but real uncertainty remains, and it's worth naming.
3. **Where the field is heading** — multimodality, voice agents, and increasingly autonomous multi-agent systems: a glimpse of what's next.
4. **Closing** — recap the day's arc, from "what is an LLM" to "how do you prove your GenAI system can be trusted," and open the floor.

## Certification quiz (15 min, end of day)

A 15-minute, 20-question, 4-choice multiple-choice quiz (Google Form) covering both presentation sessions, taken as the final activity of the day; a score of 80% or higher is required to obtain the GenAI for Finance certification from Emerton Data.
