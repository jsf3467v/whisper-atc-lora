[![CI](https://github.com/jsf3467v/whisper-atc-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/whisper-atc-lora/actions/workflows/ci.yml)

# ATC Audio-to-Text, Fine-Tuning Whisper for Air Traffic Control Transcription

This project fine-tunes Whisper-small, OpenAI's speech recognition model, using LoRA to enhance the transcription of air traffic control (ATC) radio communications. The main challenges are accurately reading callsigns and maintaining this accuracy across audio from various airports recorded at different bandwidths and noise levels compared to the training data. Three public ATC corpora are combined into one dataset, which is evaluated twice: once with a pretrained baseline and again after LoRA fine-tuning. A leak check was implemented to exclude test utterances from the training data as well, ensuring that the reported improvements are based on data without overlap. The most notable results are on ATCO2, a corpus not included in training, where the word error rate dropped from $0.559$ to $0.286$, and exact callsign matches more than doubled. Since this audio was previously unseen by the model, these improvements suggest genuine learning rather than memorization.

All training and inference run locally on Apple Silicon (M4 Max, MPS). The code is device-agnostic and runs unchanged on CUDA.

A **[live demo](https://huggingface.co/spaces/jsf3467v/atc-audio-to-text)** lets you upload or record an ATC clip and read back the transcript with its extracted callsign.

---

## Results

The table below reports Word Error Rate (WER) and callsign metrics for the pretrained baseline and the LoRA-fine-tuned model, with lower WER and higher callsign scores indicating better performance.

| Split | n | Recall | Precision | Exact | WER |
|---|---|---|---|---|---|
| in-domain ATCOSIM (clean sim) | 1901 | 0.703 to **0.984** | 0.761 to **0.979** | 0.361 to **0.930** | 0.307 to **0.021** |
| in-domain UWB-ATCC (real radio) | 2822 | 0.285 to **0.778** | 0.377 to **0.798** | 0.093 to **0.547** | 0.869 to **0.188** |
| in-domain overall | 4723 | 0.485 to **0.877** | 0.580 to **0.886** | 0.206 to **0.707** | 0.611 to **0.112** |
| in-domain leak-free | 4231 | 0.470 to **0.870** | 0.570 to **0.882** | 0.194 to **0.695** | 0.619 to **0.116** |
| **OOD ATCO2 (unseen corpus)** | 871 | 0.353 to **0.633** | 0.464 to **0.676** | 0.136 to **0.331** | 0.559 to **0.286** |

The results demonstrate that fine-tuning significantly improved the most challenging in-domain segment. The pretrained Whisper-small model failed on 8 kHz narrowband UWB-ATCC radio, achieving a 0.869 WER, but fine-tuning reduced this to 0.188, with a 4.6-fold improvement.

These enhancements are not caused by memorized data. The leak-free row, which omits overlapping utterances from the training set, has a WER of 0.116 compared to the overall 0.112, showing that in-domain improvements remain even after excluding leaked examples.

The out-of-distribution results are the most significant and remain consistent after resampling. A bootstrap analysis with 10,000 samples on ATCO2 shows an OOD WER of 0.286 within a 95% confidence interval of [0.271, 0.302]. The paired difference over the baseline is -0.272, with a confidence interval of [-0.303, -0.245]. Since this interval lies entirely below zero, the improvement is real rather than an artifact of sampling. Because the model was not trained on ATCO2, the gain reflects genuine generalization.

All metrics are based on text normalized through a shared process, eliminating formatting differences such as number words and inconsistent casing. Therefore, the reported gains reflect true recognition improvements rather than reformatting.

---

## Approach

### Data

All corpora are publicly available on the Hugging Face Hub, whereas the larger paid datasets, ATCO2-PL and LDC-ATCC, are intentionally excluded.

| Corpus | Role | Approx. size | Native rate | License |
|---|---|---|---|---|
| ATCOSIM | in-domain (clean simulation) | about 10 h | 32 kHz | per original creators |
| UWB-ATCC | in-domain (real radio) | about 20 h | 8 kHz | CC BY-NC-SA 4.0 (non-commercial) |
| ATCO2-1h | **out-of-distribution** test | about 1 h | 16 kHz | free ATCO2 subset |

The training and validation sets use data from ATCOSIM and UWB-ATCC, with the remaining data serving as the in-domain test set. ATCO2-1h is reserved exclusively for out-of-distribution testing and is not included in any other training set.

### Normalization and scoring

Before calculating any metric, a single normalizer (Datasets/normalize.py) processes both the reference and the hypothesis. Before scoring, it runs them through the same steps. It cleans and lowercases the text, strips non-speech markup, expands digit runs (290 becomes two nine zero), and standardizes ICAO phonetics so that variants such as alfa and alpha collapse to one form. Applying the same function to both sides keeps the comparison fair.


The accuracy of callsign detection depends on a structure-based extractor that recognizes the two common forms of callsigns in ATC speech. An airline callsign combines an operator name with digits or phonetic spelling, such as `delta four seven zero`, whereas a registration callsign consists of a sequence of phonetic letters, such as `november one two three alpha`. The extractor's performance is measured using token-level precision and recall for the leading part, as well as exact match and coverage metrics that indicate how often a callsign is correctly identified.

### Model and training

The main model is Whisper-small, featuring 244 million parameters. It includes a LoRA adapter of rank 16 and alpha 32, applied to attention query and value projections. Training lasts three epochs in fp32 on MPS at a learning rate of $10^{-3}$. The best checkpoint, based on validation loss, is saved. Afterwards, the adapter merges into the base weights and is exported as a standard model, ensuring downstream tasks are unaware that LoRA was used.

### Evaluation

A single scorer evaluates both the baseline and fine-tuned models, reporting results by source. This includes overall scores, leak-free in-domain metrics, and out-of-distribution performance, which is the key indicator of generalization. The scorer directly checks for data leakage. Because the CI unit tests cover both the normalizer and the callsign extractor, their definitions remain consistent as the code evolves.

---

## Repository structure

```
ATC/
├── Datasets/
│   ├── data.py            # unify the three corpora into one DatasetDict (+ resume snapshot)
│   ├── audio.py           # shared soundfile/scipy decoder, reused by every stage
│   ├── normalize.py       # shared text normalizer (refs and hyps)
│   └── inspect_data.py    # data inventory / sanity checks
├── EDA/
│   └── EDA.ipynb          # exploratory analysis answering design questions
├── SRC/
│   ├── transcribe.py      # run a Whisper model over the splits -> hypotheses (JSONL)
│   ├── train.py           # LoRA fine-tuning -> merged model
│   └── infer.py           # transcribe arbitrary audio files (transcript + callsign)
├── Evaluation/
│   ├── scoring.py         # WER + callsign metrics (the shared scorer)
│   ├── evaluate.py        # read predictions, print the before/after table
│   └── error_analysis.py  # bootstrap CIs + worst-case / callsign error dump
├── tests/
│   ├── test_normalize.py  # unit tests for the shared normalizer
│   ├── test_callsign.py   # unit tests for the callsign extractor
│   └── conftest.py        # import-path wiring so pytest finds the modules
├── models/whisper-small-lora/   # exported fine-tuned model
├── predictions/                 # saved hypotheses, one <split>-<tag>.jsonl per model
├── checkpoints/                 # training checkpoints (resumable)
├── app.py                       # Gradio demo, clip -> transcript + callsign (Hugging Face Space)
├── examples/                    # sample clips surfaced in the demo UI
├── .github/workflows/ci.yml     # CI runs ruff lint, compile, notebook check, unit tests
├── requirements.txt             # pinned dependencies (install with pip install -r)
├── References / Results / Papers
```

---

## Setup

```bash
pip install -r requirements.txt
```

Audio is decoded with `soundfile` and resampled with `scipy`, which means `torchcodec` is not needed and is best left uninstalled so the audio backends do not conflict. Device handling is automatic, using MPS in fp32 on Apple Silicon and bf16 on CUDA, with no code changes needed to move between them.

---

## Reproducing the results

```bash
# 1. Download corpora, build the unified snapshot, print the inventory
python Datasets/inspect_data.py

# 2. Baseline: pretrained Whisper-small over the test splits
python SRC/transcribe.py

# 3. Fine-tune with LoRA (writes models/whisper-small-lora/)
python SRC/train.py

# 4. Fine-tuned predictions
python -c "import sys; sys.path.insert(0,'SRC'); import transcribe; \
  transcribe.main(model_id='models/whisper-small-lora', tag='whisper-small-lora')"

# 5. Score both and read the before/after
python Evaluation/evaluate.py whisper-small
python Evaluation/evaluate.py whisper-small-lora
```

Passing `epochs=1` to `train.main` runs the whole pipeline quickly as a check before committing to the full three-epoch run.

---

## Using the model on new audio

`SRC/infer.py` runs over a single file or an entire folder of clips and writes each transcript with its extracted callsign to JSONL.

```bash
# single clip
python SRC/infer.py path/to/clip.wav

# a folder of clips
python SRC/infer.py path/to/folder
```

It uses the fine-tuned model by default and writes to `transcripts.jsonl`. WAV, FLAC, OGG, and AIFF are read directly, while MP3 and M4A need converting to WAV first.

---

## Interactive demo

`app.py` wraps the fine-tuned model in a Gradio interface for uploading or recording a clip and reading back its transcript and callsign. Since it uses the same decoding, model loading, transcription, and callsign extraction as the evaluation, the demo follows the exact process that produced the reported figures, rather than a separate one.

```bash
# from the project root
pip install -r requirements.txt   # includes gradio
python app.py
```

The initial request takes some time as the model loads, but subsequent requests are quickly served from cache. The interface automatically selects MPS, CUDA, or CPU, enabling smooth operation on the free CPU-only Hugging Face Spaces tier. Short `.wav` clips in `examples/` can be accessed with one click. The hosted version is available at [jsf3467v/atc-audio-to-text](https://huggingface.co/spaces/jsf3467v/atc-audio-to-text). Like the rest of the project, it is a non-commercial research demo, licensed under the UWB-ATCC CC BY-NC-SA 4.0 license, which governed the model training.

---

## Limitations

The callsign metric serves as a heuristic indicator rather than an absolute label. Since the corpora do not annotate all sources with callsigns, the extractor follows a fixed rule that expects callsigns to appear first. This causes it to miss pilot readbacks where the callsign appears after the instruction, such as when a clearance is echoed back with the callsign at the end. The approximately 0.95 coverage rate reflects how often a callsign can be successfully extracted.

Most remaining errors in ATCO2 are understandable upon review. A brief transmission provides limited context, occasional clips may include non-English or code-switched speech not encountered during training, and longer clips can cause the decoder to loop. Two common failure modes appear. A leading registration phonetic can be reduced to a single letter, such as "oscar" becoming "o," which lowers exact-match recall even though the tokens are otherwise correct. The model also sometimes defaults to a frequent training callsign when uncertain, a bias that surveillance-context data could mitigate.

The remaining limitations are mainly related to data rather than the method. ATCOSIM is a clean, simulated, close-talk speech dataset, so its near-perfect score offers limited insight. In contrast, UWB-ATCC and ATCO2 provide more meaningful signals regarding the challenges in in-domain and out-of-domain conditions. The real-radio data is limited, with around 20 hours for training against just an hour of out-of-domain material, and it lacks American English, since the only sizable American corpus, LDC-ATCC, is paid. The free datasets (ATCOSIM, UWB-ATCC, and ATCO2) are more accessible but still carry licensing constraints. UWB-ATCC is non-commercial under CC BY-NC-SA 4.0, which keeps it suited to research rather than deployment. The normalizer is also only partial, leaving some composite numbers, such as `one hundred`, and aircraft types, such as `three twenty`, not fully canonicalized.

---

## Future work

The simplest enhancement involves the callsign extractor, which could analyze both ends of an utterance to catch readbacks it currently overlooks. Recognition might also be boosted by biasing the decoder toward aircraft known to be in the sector, inspired by the surveillance-context strategy from the ATCO2 project. A post-ASR named-entity step could then convert the raw transcript into labeled fields for callsign, command, and value, similar to ATCO2 annotations. Most remaining improvements depend on better data, as using the paid ATCO2-PL and LDC-ATCC datasets would improve American English recognition and expand acoustic coverage. Adding a pretrained Whisper-medium row could help contextualize the fine-tuned small model. The remaining issues are acoustic; targeted augmentation for narrowband and noisy radio conditions could address the errors caused by these environments.

---

## Path to Deployment

This project is the research core of an ATC transcription system rather than a finished product. It establishes that domain-specific fine-tuning improves accuracy and generalizes to unseen audio. It is roughly a fifth of what a deployable system requires, and the rest is divided among questions of data rights, serving infrastructure, and safety engineering, which the sections below take up in turn.

### Licensing

The model inherits the UWB-ATCC CC BY-NC-SA 4.0 non-commercial license, which rules out commercial deployment as trained. Any production version would have to be retrained on commercially cleared material, whether it is the paid LDC-ATCC and ATCO2-PL sets or licensed proprietary recordings.

### Serving and streaming

The current `infer.py` processes data in batches, but a real-time deployment requires a streaming pipeline that uses voice activity detection to segment the continuous radio stream into utterances. It should perform inference on low-latency chunks and be integrated with a serving layer that handles queueing and autoscaling, all operating under real-time latency rather than processing an entire file in one pass.

### Robustness in various conditions

The model was trained on three airspace corpora, and its $0.286$ WER on ATCO2 indicates a decline in accuracy when audio departs from this distribution. Improving production performance takes more than a better model. It also requires expanding the training data, augmenting it for noisy and narrowband audio, developing methods to manage overlapping transmissions, and evaluating continuously on live traffic.

### Handling callsigns operationally

Callsign detection primarily relies on a leading-span heuristic, which has a known blind spot regarding trailing readbacks. Therefore, a deployed system should combine a dedicated callsign model with surveillance-context biasing that provides information about the aircraft currently in the sector, enabling recognition to favor the callsigns that are actually present.

### Monitoring and human factors

During operation, the system should assign a confidence score to each transcript, enabling low-confidence outputs to be flagged. It should monitor for drift and log data for feedback and retraining, while remaining in a clearly human-in-the-loop role that supports controllers or provides analytics, rather than making autonomous decisions.

### Safety and regulatory compliance

ATC systems are safety-critical, so any component operating in live environments must comply with regulatory standards such as FAA requirements, undergo formal validation and certification, and be inherently fail-safe. In the short term, this suggests keeping ASR off the critical path, where it can assist with workload analysis, training tools, post-hoc analytics, and interface pre-fill. Operational deployment requires a comprehensive certification process that goes beyond model accuracy.

---

## Data licensing and attribution

ATCOSIM and the free ATCO2-1h subset are openly available, and UWB-ATCC is released under CC BY-NC-SA 4.0 for non-commercial use. All three are used here for research only, with full citations gathered in `References`.

---

## References

### Data sources (Hugging Face Hub)

- [ATCOSIM (`Jzuluaga/atcosim_corpus`)](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus)
- [UWB-ATCC (`Jzuluaga/uwb_atcc`)](https://huggingface.co/datasets/Jzuluaga/uwb_atcc)
- [ATCO2-1h (`Jzuluaga/atco2_corpus_1h`)](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h)

### Model (Hugging Face Hub)

- [Whisper-small, OpenAI (`openai/whisper-small`)](https://huggingface.co/openai/whisper-small)

### Code

- [idiap/atco2-corpus](https://github.com/idiap/atco2-corpus), data-preparation and citation bibtex

### Papers

- [Whisper (Radford et al., 2023)](https://arxiv.org/abs/2212.04356)
- [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
- [ATCO2 corpus (Zuluaga-Gomez et al., 2023)](https://arxiv.org/abs/2211.04054)
- Wav2Vec 2.0 domain-shifted ASR benchmark on ATC (Zuluaga-Gomez et al., IEEE SLT 2022)
- Callsign detection and surveillance-based biasing, and BERTraffic (Zuluaga-Gomez et al.). See the idiap/atco2-corpus repo for citations


