# Data Licensing and Attribution

This project downloads each corpus at runtime from the Hugging Face Hub and keeps the data with its original distributor. This file documents the source, citation, and license terms for each corpus, and the obligations they place on this project and on anyone using its outputs.

The MIT `LICENSE` file covers the source code only. The datasets below and any model weights trained on them keep their own terms.

---

## ATCOSIM

- **Role here:** in-domain, clean simulated speech (~10 h).
- **Provider:** Graz University of Technology (TUG) and Eurocontrol Experimental Centre (EEC).
- **Access used:** Hugging Face `Jzuluaga/atcosim_corpus`. Original: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
- **Citation:** Hofbauer, K., Petrik, S., & Hering, H. (2008). *The ATCOSIM Corpus of Non-Prompted Clean Air Traffic Control Speech.* LREC. https://aclanthology.org/L08-1507/
- **License/terms:** Distributed free of charge for research under terms set by TUG and Eurocontrol, outside the standard Creative Commons (CC) licenses. **Verify the conditions on the SPSC page before any non-research or commercial use.**
- **Obligation:** attribute by citing the paper above.

## UWB-ATCC

- **Role here:** in-domain, real ATC radio (~13 h, 8 kHz).
- **Provider:** University of West Bohemia, Department of Cybernetics.
- **Access used:** Hugging Face `Jzuluaga/uwb_atcc`. Corpus-creation paper: Šmídl et al. (2018), *Design and Development of Speech Corpora for Air Traffic Control Training*, LREC. https://aclanthology.org/L18-1450.pdf
- **License/terms:** **CC BY-NC-SA 4.0.**
- **Obligations:**
  - **Attribution (BY):** credit the corpus and cite the source.
  - **Non-commercial (NC):** use only for non-commercial purposes.
  - **Share-alike (SA):** release any derivative under the same license. The fine-tuned model in this project is such a derivative, and its weights carry CC BY-NC-SA 4.0 terms in place of MIT.

## ATCO2-1h

- **Role here:** out-of-distribution test set (~1 h, 16 kHz), held out from training.
- **Provider:** the ATCO2 consortium (Idiap, Brno University of Technology, and partners).
- **Access used:** Hugging Face `Jzuluaga/atco2_corpus_1h`. Free subset originally from https://www.atco2.org/data
- **Citation:** Zuluaga-Gomez, J., et al. (2023). *ATCO2 corpus: A Large-Scale Dataset for Research on ASR and NLU of Air Traffic Control Communications.* arXiv:2211.04054. https://arxiv.org/abs/2211.04054
- **License/terms:** The one-hour subset is offered as a free research download. The full ATCO2 corpus (test set and the ~5,281 h PL set) is distributed commercially through ELDA and falls outside this project. **Verify the exact terms of the free subset on atco2.org before any commercial use.**
- **Obligation:** attribute by citing the paper above.

---

## Summary of obligations for this project

- **Attribute all three** corpora by citing their papers (see the References section of `README.md`).
- **Keep use non-commercial.** UWB-ATCC's NC term governs the project as a whole, since the model is trained on it.
- **Share-alike on derivatives.** The fine-tuned model weights are a derivative of UWB-ATCC and carry CC BY-NC-SA 4.0. Redistribute the weights only under those terms.
- This is a research and educational portfolio project.

A model free of these restrictions requires retraining on data whose license permits commercial use (for example, the paid LDC-ATCC corpus or other permissively licensed ATC speech), with this file updated to match.
