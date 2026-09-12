# Jonathan Locala

Computer Science at Johns Hopkins, class of 2027, with minors in applied mathematics and statistics and in entrepreneurship and management.

I build machine learning and LLM systems and take them to production. Most of my recent work falls in three places: speech models that run entirely on-device, agents that produce checkable operations instead of free text, and pipelines that turn raw sensor data into reports people actually use.

**Right now:** research assistant at the JHU AI for Surgery Lab, continuing a smart intersection safety project at JHU, and looking for new grad software engineering roles starting in 2027.

[jonathanr@locala.com](mailto:jonathanr@locala.com) · [LinkedIn](https://www.linkedin.com/in/jonathan-locala/)

---

## What I've been working on

### On-device voice for clinical software
**Medical Informatics Engineering, summer 2026**

A three-stage voice stack (wake word detection, then speaker verification, then Whisper dictation) shipped into MIE's shared React design system, [mieweb/ui](https://github.com/mieweb/ui). All inference runs in the browser through ONNX Runtime and WebAssembly in a Web Worker, so patient audio never leaves the device.

<details>
<summary>How the models got good</summary>

I trained and evaluated the wake word models on a 4x V100 cluster behind a false-alarms-per-hour testing harness, which is the metric that actually matters for something always listening. Two findings drove most of the work:

* The original models had an accent blind spot. Recall on non-American English was 11%. Rebalancing the training data and retraining brought it to over 90%.
* Cross-phrase false triggers, where one wake word fires on another, were at 56%. Tightening the threshold fork and the negative sampling cut that to 2%.

End-to-end latency stayed under 250ms throughout. I also picked this project up from a previous intern's unmerged work, so a good chunk of it was diagnosing why the existing models underperformed before writing anything new.

</details>

<details>
<summary>Shipping an LLM editing agent</summary>

Separately at MIE, I took an internal AI video editing tool from prototype to production on the company's Linux cluster, with a GitHub Actions pipeline and health-checked auto-rollback that deploys in under two minutes. It has run unattended since a customer conference.

The interesting part was the agent. Instead of having the model rewrite a transcript directly, it emits validated edit operations against an immutable transcript snapshot. Every edit is auditable and reversible, and a malformed operation fails validation rather than silently corrupting the output.

Most of the editor work itself is public in [mieweb/ui](https://github.com/mieweb/ui/commits/main/?author=jlocala1): a real undo and redo stack the editor never had, drag selection that follows the pointer past the pane and auto-scrolls at the edges, selection-wide playback speed with markers that survive a transcript rebuild, and a regression suite for the edge cases that kept biting. I also added a display-title metadata key to the resumable upload layer in [mieweb/pulsevault](https://github.com/mieweb/pulsevault) and wired it through [mieweb/pulse](https://github.com/mieweb/pulse).

</details>

### Grounding LLM findings in source data
**JHU smart intersection project, summer 2026 to present**

An end-to-end Python pipeline that turns BlueCity and Ouster LiDAR sensor data into automated traffic safety reports. It scripts the vendor's web portal with Playwright for authentication, date range selection, and multi-panel exports, then derives peak demand, turning movement, and jaywalking origin-destination metrics with pandas. A full day of manual reporting now takes about 90 seconds.

<details>
<summary>The validation layer</summary>

The reports include LLM-generated findings, which meant I needed a way to stop it from publishing things that were not in the data. The validation layer checks every generated claim against the underlying records and rejects anything unverifiable before it reaches the report.

I validated the whole pipeline end to end, from live sensor collection through a PostgreSQL ingest of over 500,000 records, behind a pytest suite and GitHub Actions CI. There is also a dependency-free browser interface built on the Python standard library so non-technical users can run the pipeline themselves, backed by a PostgreSQL evidence store that regenerates reports for any stored date range in about 19 seconds without re-collecting from the sensor.

This work contributed to a six-month analysis delivered to the Prince George's County Department of Public Works and Transportation.

</details>

### Speech and retrieval infrastructure
**LaunchStack, January to May 2026**

Engineering on an open-source agentic AI platform (790+ stars). I designed and shipped a Python speech-to-text pipeline for MP3 and MP4 inputs that produces timestamped transcripts synced to a Next.js viewer with click-to-seek audio navigation.

<details>
<summary>Two backends and a retrieval layer</summary>

I implemented two transcription backends with deliberately different tradeoffs: a Python service running OpenAI Whisper in a sidecar, and a JavaScript sherpa-onnx version running in-process for people who want a simpler self-hosted deployment. On top of that, a multi-model routing system that switches between OpenAI and Anthropic depending on the task.

I also built the document ingestion and metadata extraction pipeline over a PostgreSQL and pgvector retrieval layer, parsing unstructured uploads into structured outputs that feed downstream agentic workflows.

</details>

### Predicting the 2026 World Cup
**Independent project**

[WorldCupPredictor](https://github.com/jlocala1/WorldCupPredictor) forecasts match outcomes and simulates the tournament. Three calibrated models (logistic regression, random forest, XGBoost) vote in an ensemble over 62 engineered features covering form, head-to-head history, Elo, FIFA rankings, and positional metrics, with a four-tier cascading scraper across Understat, Transfermarkt, fotmob, and fbref so a missing source degrades instead of breaking the run.

<details>
<summary>Simulation and evaluation</summary>

Tournament outcomes come from a Monte Carlo simulation that respects FIFA's actual 2026 bracket structure, including penalty shootout modeling. Training runs on 2010 to 2021, validation on 2022 to 2024, and the held-out test set is 2025 through early 2026, which keeps the evaluation honest about forecasting forward rather than fitting the past.

Test log loss came in at 0.827, benchmarked against bookmaker and FiveThirtyEight baselines, and I ran an LLM as an additional baseline to see how a general model does at the same task. Ablations and hyperparameter tuning are in the repo, along with the known limitations.

</details>

### Sports analytics research
**JHU Sports Analytics Research Group, fall 2024 to May 2026**

The project I am proudest of is an Expected Possession Value framework for soccer. Random Forest models trained on tracking and event data from 500+ MLS matches recursively evaluate shot, pass, and dribble decisions through Bellman-style action-value calculations, personalized across 1,000+ individual players.

<details>
<summary>EPV, field goals, and defender ratings</summary>

For EPV, I engineered the spatiotemporal features that give the model game context: a pitch control model, defensive positioning metrics, and passing lane analysis. The recursive formulation is what lets it value a decision by what it makes possible next rather than only by what immediately happened.

Two other projects from the same group:

* **Field goal prediction for the Cleveland Browns.** XGBoost models at over 90% accuracy on a Trackman dataset of 3,800+ kicks covering direction, speed, height, distance, ball rotation, and launch angle. I worked directly with the Browns' Research and Strategy team on player evaluation and kicking strategy.
* **Defender evaluation dashboard** for 2,000+ players across Europe's top five leagues, using XGBoost with PCA and Optuna hyperparameter optimization, shipped as an interactive Tableau dashboard with position filtering and multi-season comparison.

</details>

### Incoming: AI for Surgery Lab
**JHU, fall 2026**

Joining a lab that builds machine learning systems to assess operative technique from surgical video and motion data, generating the kind of coaching feedback an experienced surgeon would give a trainee.

---

## Tech

**ML and AI:** PyTorch (GPU/CUDA), Hugging Face, OpenAI and Anthropic APIs, LangChain, OpenAI Whisper, sherpa-onnx and ONNX Runtime, scikit-learn, XGBoost, Optuna

**Data:** PostgreSQL (pgvector), pandas, NumPy, Matplotlib, Tableau

**Languages:** Python, TypeScript, JavaScript, R, Java, C++, C, SQL, HTML/CSS

**Web and tools:** Next.js, React, Node.js, REST APIs, Docker, Git, GitHub Actions, Playwright, pytest, WebAssembly, ffmpeg, Unix/Linux

---

## Outside of code

I played on the Johns Hopkins men's soccer team, NCAA Division III, and was part of the 2023 and 2024 Centennial Conference championship sides. Dean's List student-athlete and 2024 Centennial Conference Academic Honor Roll. Vice President of Phi Kappa Psi. I also play jazz.
