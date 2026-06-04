# Data Licensing and Attribution

This project does not redistribute any dataset. The code downloads the corpora at
runtime from the Hugging Face Hub. This file documents the source, citation, and
license terms for each, and the obligations they place on this project and on
anyone using its outputs.

The `LICENSE` (MIT) file covers the source code only. It does **not** cover the
datasets below or any model weights trained on them.

---

## ATCOSIM

- **Role here:** in-domain, clean simulated speech (~10 h).
- **Provider:** Graz University of Technology (TUG) and Eurocontrol Experimental Centre (EEC).
- **Access used:** Hugging Face `Jzuluaga/atcosim_corpus`. Original: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
- **Citation:** Hofbauer, K., Petrik, S., & Hering, H. (2008). *The ATCOSIM Corpus of Non-Prompted Clean Air Traffic Control Speech.* LREC. https://aclanthology.org/L08-1507/
- **License/terms:** Distributed free of charge for research. ATCOSIM does not carry a
  standard CC tag; its terms are set by TUG/Eurocontrol. **Verify the conditions on
  the SPSC page before any non-research or commercial use.**
- **Obligation:** attribute by citing the paper above.

## UWB-ATCC

- **Role here:** in-domain, real ATC radio (~20 h, 8 kHz).
- **Provider:** University of West Bohemia, Department of Cybernetics.
- **Access used:** Hugging Face `Jzuluaga/uwb_atcc`. Corpus-creation paper: Šmídl et al. (2018), *Design and Development of Speech Corpora for Air Traffic Control Training*, LREC. https://aclanthology.org/L18-1450.pdf
- **License/terms:** **CC BY-NC-SA 4.0.**
- **Obligations:**
  - **Attribution (BY):** credit the corpus and cite the source.
  - **Non-commercial (NC):** use only for non-commercial purposes.
  - **Share-alike (SA):** any derivative must be released under the same license.
    The fine-tuned model in this project is such a derivative, which is why its
    weights are **not** MIT-licensed and inherit CC BY-NC-SA 4.0 terms.

## ATCO2-1h

- **Role here:** out-of-distribution test set (~1 h, 16 kHz), never used in training.
- **Provider:** the ATCO2 consortium (Idiap, Brno University of Technology, and partners).
- **Access used:** Hugging Face `Jzuluaga/atco2_corpus_1h`. Free subset originally from https://www.atco2.org/data
- **Citation:** Zuluaga-Gomez, J., et al. (2023). *ATCO2 corpus: A Large-Scale Dataset for Research on ASR and NLU of Air Traffic Control Communications.* arXiv:2211.04054. https://arxiv.org/abs/2211.04054
- **License/terms:** The one-hour subset is offered for free download for research.
  The full ATCO2 corpus (test set and the ~5,281 h PL set) is distributed
  commercially through ELDA and is **not** used here. **Verify the exact terms of
  the free subset on atco2.org before any commercial use.**
- **Obligation:** attribute by citing the paper above.

---

## Summary of obligations for this project

- **Attribute all three** corpora by citing their papers (see `References`).
- **Keep use non-commercial.** UWB-ATCC's NC term governs the project as a whole,
  since the model is trained on it.
- **Share-alike on derivatives.** The fine-tuned model weights are a derivative of
  UWB-ATCC and therefore carry CC BY-NC-SA 4.0, not MIT. Do not redistribute the
  weights under a permissive license.
- This is a research and educational portfolio project, not a deployable commercial
  system.

If you need a model free of these restrictions, retrain using only data whose
license permits commercial use (for example, the paid LDC-ATCC corpus or other
permissively licensed ATC speech), and update this file accordingly.