[![CI](https://github.com/jsf3467v/whisper-atc-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/whisper-atc-lora/actions/workflows/ci.yml)

# Fine-Tuning Whisper for Air Traffic Control Transcription

This project fine-tunes Whisper-small, the OpenAI speech recognition model, with low-rank adaptation (LoRA) to transcribe air traffic control (ATC) radio and extract the aircraft callsign. Three public ATC corpora are unified into one dataset. Two of them supply training data, and the third, ATCO2, is held out entirely as an out-of-distribution test set. The pretrained model and the fine-tuned model are scored by the same code on the same test utterances, with confidence intervals that resample whole recordings.

On ATCO2, held out from training, word error rate (WER) fell from $0.567$ to $0.296$. The paired change is $-0.271$ with a 95% confidence interval of $[-0.299, -0.244]$. Callsign F1 on the same corpus rose from $0.111$ to $0.320$.

A **[live demo](https://huggingface.co/spaces/jsf3467v/atc-audio-to-text)** transcribes an uploaded or recorded clip and extracts its callsign. The fine-tuned weights are published as **[jsf3467v/whisper-small-atc-lora](https://huggingface.co/jsf3467v/whisper-small-atc-lora)**.

---

## Results

Word error rate counts substituted, deleted, and inserted words relative to the length of the reference. Lower is better, and the value can exceed $1$ when a model inserts many words. Each interval is a 95% bootstrap interval over 10,000 resamples of whole recordings within each corpus.

| Test split | Utterances | Pretrained WER | Fine-tuned WER | Change [95% interval] |
|---|---|---|---|---|
| ATCOSIM (in-domain, simulated) | 1,901 | 0.299 | 0.021 | $-0.278$ [$-0.297$, $-0.259$] |
| UWB-ATCC (in-domain, radio) | 2,822 | 0.834 | 0.189 | $-0.646$ [$-0.681$, $-0.613$] |
| In-domain overall | 4,723 | 0.589 | 0.112 | $-0.477$ [$-0.501$, $-0.456$] |
| In-domain, transcripts unseen in training | 4,222 | 0.597 | 0.115 | $-0.482$ [$-0.510$, $-0.457$] |
| In-domain, recordings unseen in training | 4,601 | 0.597 | 0.115 | $-0.482$ [$-0.504$, $-0.462$] |
| **ATCO2 (out-of-distribution)** | **871** | **0.567** | **0.296** | $-0.271$ [$-0.299$, $-0.244$] |

The out-of-distribution change is the primary result. Every one of the 10,000 resamples showed an improvement, and because ATCO2 was held out of training, the gain reflects generalization to recordings and recording conditions outside the training data, including Australian traffic absent from both training corpora. The largest in-domain gain is on UWB-ATCC, where the pretrained model struggled with 8 kHz narrowband radio and fine-tuning reduced word error rate by a factor of $4.4$.

The improvement holds beyond memorized transcripts. Some in-domain test transcripts also occur in the training set, and some test utterances come from recordings that also contributed training utterances. The two filtered rows remove each kind of overlap, and their error rate of $0.115$ is close to the overall $0.112$.

### Callsign recognition

The callsign is the aircraft identity, such as `csa one delta zulu`, and it is the part of a transmission that matters most operationally. Callsigns are scored as whole entities, and a prediction counts only when every token matches the callsign found in the reference.

| Test split | Pretrained F1 | Fine-tuned F1 | Fine-tuned precision | Fine-tuned recall |
|---|---|---|---|---|
| ATCOSIM | 0.351 | 0.937 | 0.935 | 0.939 |
| UWB-ATCC | 0.053 | 0.631 | 0.627 | 0.635 |
| In-domain overall | 0.204 | 0.771 | 0.767 | 0.774 |
| ATCO2 | 0.111 | 0.320 | 0.345 | 0.299 |

Fine-tuning improved callsign F1 on every split, and each improvement appeared in all 10,000 resamples. On ATCO2 the change is $+0.209$ with an interval of $[+0.171, +0.249]$.

### Error profile

On ATCO2, fine-tuning halved substitutions, from $0.370$ to $0.181$ errors per reference word, and cut insertions from $0.136$ to $0.041$. Deletions rose slightly, from $0.061$ to $0.074$, meaning the fine-tuned model occasionally drops words that the pretrained model kept. Short clips remain the hardest. ATCO2 clips under 2.8 seconds have a word error rate of $0.473$, against $0.240$ for clips of 4.6 seconds or longer, consistent with the limited context of a brief transmission.

The fine-tuned model missed 502 of the 716 ATCO2 reference callsigns. The misses fall into recognizable patterns. Registrations are mistaken for place or airline names, as when `hotel bravo kilo golf november` becomes `prague kilo golf november`. Some transmissions mix in other languages, and recordings from Sydney speak numbers in groups, as in `qantas seventy five twenty eight`. One of the 871 outputs repeated a word in a loop before recovering.

---

## Methodology

### Data

All three corpora are publicly available on the Hugging Face Hub. The larger paid corpora, ATCO2-PL and LDC-ATCC, fall outside the scope of this project.

| Corpus | Role | Train | Validation | Test | Native sample rate |
|---|---|---|---|---|---|
| ATCOSIM | in-domain, simulated close-talk speech | 7,419 | 219 | 1,901 | 32 kHz |
| UWB-ATCC | in-domain, real radio | 10,727 | 564 | 2,822 | 8 kHz |
| ATCO2-1h | out-of-distribution test only | 0 | 0 | 871 | 16 kHz |

The training split totals 18.2 hours and the in-domain test split 4.5 hours. ATCO2 contributes about one hour. The in-domain test utterances are the test splits published with each corpus. Validation is drawn from each training corpus separately, and whole recordings or speaker sessions are held out together up to about 5% of the utterances. Each recording therefore falls entirely in training or entirely in validation. The ATCOSIM share falls below 5% because its speaker sessions are large and each session stays whole.

### Text normalization

One normalizer, in `Datasets/normalize.py`, is applied to training labels, references, and model outputs alike. It lowercases the text, removes bracketed markup, spells out digit runs (290 becomes `two nine zero`), writes decimals as `point`, and maps spelling variants from the International Civil Aviation Organization (ICAO) alphabet to one form, turning `alfa` and `niner` into `alpha` and `nine`. Both sides of every comparison pass through the same function, which keeps formatting differences out of the recognition errors.

Grouped numbers such as `one hundred` keep their spoken form. A second word error rate, reported as `wer_numeric` in the results file, first converts grouped numbers to digits on both sides. On every test split it differs from the main word error rate by at most $0.007$ for the fine-tuned model and $0.035$ for the pretrained model.

### Model and training

The base model is Whisper-small, with 242 million parameters. LoRA adapters with rank $r = 16$, scaling $\alpha = 32$, and dropout $0.05$ are attached to the query and value projections of every attention layer, which leaves 1.77 million trainable parameters, 0.73% of the model. Training ran for three epochs with a batch size of 16, a learning rate of $10^{-3}$, 50 warmup steps, and linear decay. Validation loss was measured three times per epoch, and the checkpoint with the lowest validation loss was kept. It fell at every measurement, from $0.495$ to $0.280$, and the final checkpoint was therefore the best. After training, the adapters are merged into the base weights, and the result is saved as a standard Whisper model.

Training ran in 32-bit precision on an Apple M4 Max through Metal Performance Shaders and took 11.4 hours. The code selects bfloat16 or float16 precision on CUDA devices and falls back to the CPU. Checkpoints are stored under a name derived from the data code and the training settings, and an interrupted run resumes only with identical settings. If memory runs out, training halves the batch size, doubles gradient accumulation to keep the effective batch size, and resumes.

Decoding stops after 74 new tokens, the length of the longest transcript in the training split plus its end token, which bounds runaway repetition.

### Evaluation

Every metric is a corpus-level ratio, computed as the summed errors over the summed reference length across all utterances. References that normalize to an empty string are excluded. Confidence intervals come from a bootstrap that resamples whole recordings within each corpus (Bisani and Ney, 2004), which keeps correlated utterances from one recording together and holds the mix of corpora fixed. Each slice draws its own resamples, and the pretrained and fine-tuned models are scored on the same resamples within a slice, which gives paired differences and the probability of improvement. The out-of-distribution word error rate is designated as the primary endpoint, and all other comparisons are exploratory and unadjusted for multiple testing.

Callsigns are found by a rule-based extractor in `Evaluation/scoring.py`, applied identically to references and outputs. It first looks at the start of a transmission for an optional operator name of up to two words followed by digits or phonetic letters, after skipping greetings and a named station. Failing that, it checks the end of the transmission, where pilots repeat their callsign in readbacks. A tail counts as a callsign when it is an operator name followed by digits or letters, or a registration of at least three phonetic letters. Values after words such as `runway` or `squawk` and taxi routes after `via` are rejected. The reference callsign is therefore the one the extractor finds in the reference transcript, in place of a human label. It finds a callsign in 74% of in-domain references and 82% of ATCO2 references.

---

## Repository structure

```
├── Datasets/
│   ├── data.py            # unifies the corpora, recording-level validation split, cached snapshot
│   ├── audio.py           # shared audio decoding and resampling to 16 kHz
│   ├── normalize.py       # shared text normalizer
│   └── inspect_data.py    # split sizes, hours, and normalized samples
├── EDA/
│   └── EDA.ipynb          # exploratory analysis behind the design decisions
├── src/
│   ├── train.py           # LoRA fine-tuning and merge
│   ├── transcribe.py      # transcribes the test splits, model loading, decoding
│   └── infer.py           # transcribes any audio file or folder with callsigns
├── Evaluation/
│   ├── scoring.py         # metrics, callsign extractor, and bootstrap shared by all evaluation
│   ├── evaluate.py        # slices, intervals, and paired comparison
│   └── error_analysis.py  # worst utterances, callsign misses, false alarms, repetitions
├── tests/                 # unit tests for every module, with synthetic data and stand-in models
├── results/               # evaluation and error analysis output (JSON)
├── examples/              # sample clips for the demo
├── app.py                 # Gradio demo
├── requirements.txt       # pinned dependencies
├── ruff.toml              # lint and format settings
├── .github/workflows/     # continuous integration (lint, tests, compile, notebook check)
├── DATA_LICENSE.md        # corpus sources, citations, and license terms
└── LICENSE                # MIT license for the code
```

The folders `models/`, `predictions/`, `checkpoints/`, `Datasets/cache/`, and `Datasets/processed/` are created when the pipeline runs and are listed in `.gitignore`.

---

## Setup

```bash
pip install -r requirements.txt
```

The project targets Python 3.11. Audio is decoded with `soundfile` and resampled with `scipy`, which removes the need for `torchcodec`. The device is chosen automatically, in the order Apple Metal Performance Shaders, CUDA, and CPU.

---

## Reproducing the results

```bash
# 1. Download the corpora, build the data snapshot, and print the inventory
python Datasets/inspect_data.py

# 2. Transcribe the test splits with the pretrained model
python src/transcribe.py

# 3. Fine-tune with LoRA and save the merged model to models/whisper-small-lora/
python src/train.py

# 4. Transcribe the test splits with the fine-tuned model
python src/transcribe.py models/whisper-small-lora

# 5. Score both models and compare them
python Evaluation/evaluate.py whisper-small-lora whisper-small
python Evaluation/error_analysis.py whisper-small-lora

# 6. Run the unit tests and the notebook
python -m pytest tests -q
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 EDA/EDA.ipynb
```

Transcription and training both resume where they stopped if a run is interrupted. Evaluation writes `results/eval-whisper-small-lora.json`, which holds every metric for every slice, with intervals and paired changes. Error analysis writes `results/error-test_ood-whisper-small-lora.json`.

---

## Using the model on new audio

```bash
# a single clip
python src/infer.py path/to/clip.wav

# a folder of clips
python src/infer.py path/to/folder
```

The script uses the fine-tuned model by default and writes each transcript with its callsign to `predictions/transcripts.jsonl`. A different model folder and output path can be given as the second and third arguments. WAV, FLAC, OGG, MP3, and AIFF files are read directly.

---

## Interactive demo

```bash
python app.py
```

The Gradio demo uses the same model loading, decoding, and callsign extraction as the evaluation. It loads the local model by default, and the environment variable `ATC_MODEL` can point it to another folder or a Hugging Face model repository. The hosted version at [jsf3467v/atc-audio-to-text](https://huggingface.co/spaces/jsf3467v/atc-audio-to-text) runs on the free CPU tier and loads the published model. Its first request after a restart takes longer while the model loads.

---

## Limitations

The callsign extractor is a heuristic. It finds callsigns only at the start or the end of a transmission, and it occasionally mistakes a phrase for an operator name, for example a greeting in another language. Callsign scores therefore measure agreement between two applications of the same rule, in place of agreement with human labels. Under this definition the fine-tuned model also produces slightly more callsigns on UWB-ATCC references that lack one, a rate change of $+0.044$ with an interval of $[+0.011, +0.077]$.

The data sets the main boundary on these results. ATCOSIM is clean simulated speech, and its near-perfect score says little about real radio. The out-of-distribution set holds about one hour of audio, and the corpora cover European and Australian airports only, because the main American corpus, LDC-ATCC, is proprietary. Whisper reads 30-second windows and truncates longer transmissions, and every test clip fits within that window. The UWB-ATCC license is non-commercial, and the fine-tuned model inherits that restriction.

---

## Future work

Several directions follow from the error analysis. Biasing recognition toward the aircraft actually present in a sector, as in the surveillance-context approach of the ATCO2 project, would address callsigns replaced by common training callsigns. A named-entity step after recognition could label callsigns, commands, and values directly instead of relying on position. Commercially licensed data such as LDC-ATCC and ATCO2-PL would add American English and wider acoustic coverage, and targeted augmentation for narrowband and noisy audio could reduce the remaining acoustic errors. Evaluating a pretrained Whisper-medium model would measure how much of the gain comes from model size before any fine-tuning.

---

## Path to deployment

This project is the research core of an ATC transcription system. A deployed system would need commercially cleared training data, since the current model inherits a non-commercial license. It would also need a streaming pipeline that segments the continuous radio channel with voice activity detection and transcribes short chunks with low latency, in place of the current file-based batch processing. Each transcript would carry a confidence score, allowing uncertain outputs to be flagged for review, and the system would be monitored for drift on live traffic.

Because ATC is safety-critical, any component used in live operations must meet regulatory requirements, such as those of the Federal Aviation Administration, and pass formal validation. Near-term uses should therefore stay off the critical path, in roles such as workload analysis, training tools, post-event review, and pre-filling controller interfaces, with a human making every operational decision.

---

## Data licensing

The code is released under the MIT license. The corpora remain with their original distributors, and each keeps its own terms. UWB-ATCC is licensed under CC BY-NC-SA 4.0, which makes the fine-tuned model a non-commercial derivative under the same terms. The file `DATA_LICENSE.md` lists the source, citation, and terms for each corpus.

---

## References

### Data

- [ATCOSIM (`Jzuluaga/atcosim_corpus`)](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus). Hofbauer, Petrik, and Hering (2008), [The ATCOSIM Corpus of Non-Prompted Clean Air Traffic Control Speech](https://aclanthology.org/L08-1507/), LREC.
- [UWB-ATCC (`Jzuluaga/uwb_atcc`)](https://huggingface.co/datasets/Jzuluaga/uwb_atcc). Šmídl et al. (2018), [Design and Development of Speech Corpora for Air Traffic Control Training](https://aclanthology.org/L18-1450.pdf), LREC.
- [ATCO2-1h (`Jzuluaga/atco2_corpus_1h`)](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h). Zuluaga-Gomez et al. (2023), [ATCO2 Corpus](https://arxiv.org/abs/2211.04054).

### Models

- [Whisper-small (`openai/whisper-small`)](https://huggingface.co/openai/whisper-small)
- [Fine-tuned model (`jsf3467v/whisper-small-atc-lora`)](https://huggingface.co/jsf3467v/whisper-small-atc-lora)

### Methods

- Radford et al. (2023), [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356).
- Hu et al. (2022), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Bisani and Ney (2004), Bootstrap Estimates for Confidence Intervals in ASR Performance Evaluation, ICASSP.
- [idiap/atco2-corpus](https://github.com/idiap/atco2-corpus), data preparation and citations for the ATCO2 project, including callsign detection and surveillance-context biasing.
