# ATC Audio-to-Text: Fine-Tuning Whisper for Air Traffic Control Transcription

This project fine-tunes OpenAI's Whisper-small model using LoRA to transcribe air traffic control (ATC)
radio communications, with a special focus on **callsign accuracy** and
**out-of-distribution generalization**. It processes raw public ATC datasets through a complete pipeline that includes data unification, exploratory analysis, a zero-shot
baseline, parameter-efficient fine-tuning, and leak-checked evaluation. The results show that training on existing ATC speech enables the model to perform well on a corpus it has not encountered before.


All training and inference run locally on Apple Silicon (M4 MAX, MPS); the code is
device-agnostic and runs unchanged on CUDA.

**[Try the live demo](https://huggingface.co/spaces/jsf3467v/atc-audio-to-text)** — upload or
record an ATC clip and read back the transcript with its extracted callsign.

---

## Results

Word Error Rate (WER) and callsign metrics, zero-shot baseline -> LoRA fine-tuned.
Lower WER is better; higher callsign scores are better.

| Split | n | Recall | Precision | Exact | WER |
|---|---|---|---|---|---|
| in-domain ATCOSIM (clean sim) | 1901 | 0.703 → **0.984** | 0.761 → **0.979** | 0.361 → **0.930** | 0.307 → **0.021** |
| in-domain UWB-ATCC (real radio) | 2822 | 0.285 → **0.778** | 0.377 → **0.798** | 0.093 → **0.547** | 0.869 → **0.188** |
| in-domain overall | 4723 | 0.485 → **0.877** | 0.580 → **0.886** | 0.206 → **0.707** | 0.611 → **0.112** |
| in-domain leak-free | 4231 | 0.470 → **0.870** | 0.570 → **0.882** | 0.194 → **0.695** | 0.619 → **0.116** |
| **OOD ATCO2 (unseen corpus)** | 871 | 0.353 → **0.633** | 0.464 → **0.676** | 0.136 → **0.331** | 0.559 → **0.286** |

Three things to read from this table:

- **The hardest in-domain slice was rescued.** A basic Whisper-small effectively
  failed on 8 kHz narrowband UWB-ATCC radio (0.869 WER). Fine-tuning cut that to
  0.188, a 4.6× reduction.


- **The gains are not memorization.** The leak-free row (training-overlapping
  utterances removed) is essentially identical to the overall row (0.112 vs 0.116
  WER), so the in-domain improvement survives when leaked examples are stripped out.


- **The model demonstrates strong generalization.** On ATCO2, a dataset it was not trained on and from different airports 
the Word Error Rate (WER) nearly halved (from 0.559 to 0.286), and the callsign exact-match more than doubled. This is the key result, 
as performance on out-of-distribution data cannot be attributed to fitting the training data.

WER and callsign metrics are computed on text passed through a shared normalizer, so
formatting differences (digits-as-words, casing) are neutralized on both sides and
the improvements reflect genuine recognition, not output reformatting.

---

## Approach

### Data

All corpora are freely available and accessible through the Hugging Face Hub. The large paid datasets, ATCO2-PL
set and LDC-ATCC, are intentionally excluded.


| Corpus | Role | Approx. size | Native rate | License |
|---|---|---|---|---|
| ATCOSIM | in-domain (clean simulation) | ~10 h | 32 kHz | per original creators |
| UWB-ATCC | in-domain (real radio) | ~20 h | 8 kHz | CC BY-NC-SA 4.0 (non-commercial) |
| ATCO2-1h | **out-of-distribution** test | ~1 h | 16 kHz | free ATCO2 subset |

Splits: `train` and `validation` are drawn from ATCOSIM + UWB-ATCC; `test_indomain`
is their held-out test sets; `test_ood` is all of ATCO2-1h, never trained on.

### Normalization and scoring

A single normalizer (`Datasets/normalize.py`) is consistently applied to both references
and hypotheses before calculating any metric. This process includes lowercasing, removing non-speech markup, expanding digit runs (`290` becomes `two nine zero`), and ICAO phonetic
canonicalization (`alfa`/`alpha` becomes `alpha`). Using the same normalization function throughout ensures that all evaluations are comparable.


Callsign accuracy uses a structure-based extractor that handles both ATC callsign formats: 
the operator name followed by digits or phonetic spelling (like "delta four seven zero" for 
airline style), or a sequence of phonetic letters (such as "hotel golf echo" for registration style). 
This is evaluated with token precision/recall (for the lead), exact match,
and coverage metrics.


### Model and training

Whisper-small (244 M parameters) with LoRA adapters (rank 16, α 32, on the attention
query/value projections), trained for 3 epochs at learning rate 1e-3 in fp32 on MPS.
The best checkpoint is selected by validation loss, then the adapter is merged into
the base weights and exported as a standard model directory.

### Evaluation

Evaluation is entirely independent of training. The same scorer assesses both the zero-shot
baseline and the fine-tuned model, providing reports for per-source, overall, leak-free
in-domain, and OOD results. In-domain leakage is explicitly verified, while OOD is considered a reliable indicator of generalization.


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
│   └── evaluate.py        # read predictions, print the before/after table
├── models/whisper-small-lora/   # exported fine-tuned model
├── predictions/                 # saved hypotheses, one <split>-<tag>.jsonl per model
├── checkpoints/                 # training checkpoints (resumable)
├── app.py                       # Gradio demo: clip -> transcript + callsign (Hugging Face Space)
├── examples/                    # sample clips surfaced in the demo UI
├── requirements.txt             # pinned dependencies (install with pip install -r)
├── References / Results / Papers
```


---

## Setup

```bash
pip install -r requirements.txt
```

Notes:
- Audio is decoded with `soundfile` and resampled with `scipy`, so `torchcodec` is
  **not** required (and is best left uninstalled to avoid backend conflicts).
- On Apple Silicon the code uses MPS in fp32; on CUDA it uses bf16. No edits needed
  to switch.

---

## Reproducing the results

```bash
# 1. Download corpora, build the unified snapshot, print the inventory
python Datasets/inspect_data.py

# 2. Baseline: zero-shot Whisper-small over the test splits
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

A first run can use `train.main(epochs=1)` to validate the full pipeline quickly
before committing to the longer three-epoch run.

---

## Using the model on new audio

`SRC/infer.py` transcribes a single file or a folder of clips and writes the
transcript and extracted callsign to JSONL:

```bash
# single clip
python SRC/infer.py path/to/clip.wav

# a folder of clips
python SRC/infer.py path/to/folder
```

Defaults to the fine-tuned model and writes `transcripts.jsonl`. WAV/FLAC/OGG/AIFF
are read directly; MP3/M4A should be converted to WAV first.

---

## Interactive demo

`app.py` wraps the fine-tuned model in a small Gradio interface: upload or record a clip and
read back the transcript and the extracted callsign. It reuses the same decode, model loading,
transcription, and callsign extraction as the evaluation (`transcribe` and `scoring`), so the
demo runs the exact path the numbers above score rather than a separate one.

```bash
# from the project root
pip install -r requirements.txt   # includes gradio
python app.py
```

The first request loads the model and is slow; every request after is cached. The interface
selects MPS, CUDA, or CPU automatically, so it runs unchanged on the free CPU-only Hugging Face
Spaces tier. Drop a few short `.wav` clips in `examples/` and they appear as one-click samples.

A hosted version runs as a Hugging Face Space:
[jsf3467v/atc-audio-to-text](https://huggingface.co/spaces/jsf3467v/atc-audio-to-text). Like the
rest of the project it is a non-commercial research demo, consistent with the UWB-ATCC
CC BY-NC-SA 4.0 license under which the model was trained.

---

## Limitations

- **The callsign metric serves as a heuristic proxy rather than a definitive label.** 
Since the corpora do not provide callsign annotations for every source, the extractor 
relies on a fixed rule to read the reference text. It assumes the callsign appears first, 
which causes it to miss pilot *readbacks* where the callsign comes after the instruction 
(e.g., "descending flight level one hundred ... csa one delta zulu" yields an incorrect span). 
Coverage (~0.95) indicates how often a callsign can be extracted.


- **ATCOSIM's near-perfect score reflects an easier scenario.** It involves clean, simulated, 
close-talk speech; the meaningful signals are UWB-ATCC (challenging in-domain) and ATCO2 (out-of-domain).


- **Limited real-radio data** — about 20 hours of real ATC recordings for training and 1 hour out-of-domain.


- **The data is European in origin.** No large free American ATC corpus is available (LDC-ATCC is paid); 
while code examples use American conventions, the actual audio does not.


- **UWB-ATCC is non-commercial (CC BY-NC-SA 4.0),** making this a research or portfolio project rather 
than a deployable commercial system.

- **Partial number normalization** — composite values like "one hundred" or aircraft types such as "three twenty" are not fully canonicalized.


---

## Future work

- **Callsign extraction:** handle trailing/readback callsigns (check both ends of the
  utterance), improving the metric on pilot transmissions.


- **Callsign biasing:** feed nearby aircraft callsigns from surveillance context into
  decoding, following the ATCO2 line of work, to boost callsign recognition.


- **Structured extraction:** a post-ASR NER stage for callsign / command / value, as
  in the ATCO2 corpus annotations.


- **More and broader data:** the paid ATCO2-PL and LDC-ATCC sets, including American
  English, to widen acoustic and accent coverage.


- **Model size context:** a Whisper-medium zero-shot row alongside the fine-tuned
  small model, reported as context rather than as the comparison.


- **Acoustic robustness:** augmentation targeting narrowband/noisy radio, where the
  largest error remains.

---

## Path to Deployment

This project serves as the research core of an ATC transcription system—proof that domain-specific fine-tuning is effective and generalizes well. It is not a complete, deployable product. The model represents about 20% of a full system; the remaining components include data rights, infrastructure, and safety engineering. To turn this into a deployable system, the following steps are necessary:

### Licensing 
The model carries UWB-ATCC's CC BY-NC-SA 4.0 (non-commercial) license, which prohibits commercial deployment as trained. A production system must be retrained using commercially cleared data—such as paid datasets from LDC-ATCC or ATCO2-PL, or licensed proprietary recordings.

### Serving and streaming
`infer.py` conducts batch transcription, but for live setups, a streaming pipeline is essential. This pipeline should include voice activity detection to segment continuous radio signals into utterances, enable low-latency, chunked inference, and feature a serving layer with an API, queue, and autoscaling. It must also meet real-time latency targets instead of just performing a single process.

### Robustness in various conditions
The model was trained on three specific airspace corpora, and the measured out-of-distribution (OOD) gap (0.286 WER on ATCO2) indicates accuracy declines outside the training data. A production system would require wider training data, augmentation for noise and narrowband environments, handling of overlapping transmissions, and continuous evaluation using real traffic.

### Handling callsigns operationally
Currently, callsign detection relies on a leading-span heuristic, which has a known blind spot with trailing readbacks. Deploying a system would need a dedicated callsign extraction model and, importantly, surveillance-context biasing—supplying the list of aircraft in the sector so that recognition emphasizes actual callsigns.

### Monitoring and human factors
The system should include confidence scores for each transcript to flag low-confidence outputs, monitor for drift, log data for feedback and retraining, and have a clear human-in-the-loop role—such as assisting controllers or providing analytics, not decision-making.

### Safety and regulatory compliance
ATC is a safety-critical system. All systems involved in live operations must comply with regulatory standards (FAA/EASA), 
go through formal validation and certification processes, and include fail-safe features. In the near term, efforts should 
prioritize these elements to ensure safe deployment. Keep ASR off the critical path, in roles such as workload analysis, 
training tools, post-hoc analytics, and interface pre-fill. Direct operational use involves a certification process that 
extends well beyond model accuracy.

---

## Data licensing and attribution

ATCOSIM and the free ATCO2-1h subset are openly available; UWB-ATCC is licensed
CC BY-NC-SA 4.0 (non-commercial). The corpora are used here for research only.
Full citations are in `References`.

---

## References

### Data sources (Hugging Face Hub)

- ATCOSIM (`Jzuluaga/atcosim_corpus`): https://huggingface.co/datasets/Jzuluaga/atcosim_corpus
- UWB-ATCC (`Jzuluaga/uwb_atcc`): https://huggingface.co/datasets/Jzuluaga/uwb_atcc
- ATCO2-1h (`Jzuluaga/atco2_corpus_1h`): https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h

### Model (Hugging Face Hub)

- Whisper-small, OpenAI (`openai/whisper-small`): https://huggingface.co/openai/whisper-small

### Code

- idiap/atco2-corpus, data-preparation recipes and citation bibtex: https://github.com/idiap/atco2-corpus

### Papers

- Whisper (Radford et al., 2023): https://arxiv.org/abs/2212.04356
- LoRA (Hu et al., 2022): https://arxiv.org/abs/2106.09685
- ATCO2 corpus (Zuluaga-Gomez et al., 2023): https://arxiv.org/abs/2211.04054
- Wav2Vec 2.0 domain-shifted ASR benchmark on ATC (Zuluaga-Gomez et al., IEEE SLT 2022)
- Callsign detection and surveillance-based biasing, and BERTraffic (Zuluaga-Gomez et al.); see the idiap/atco2-corpus repo for citations