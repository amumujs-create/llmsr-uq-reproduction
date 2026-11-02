# Sources and licenses

This is an independent reproduction harness, not the authors' official repository.

## Original implementation

- Paper: [Know When the Equation Is Wrong: Structural Uncertainty for LLM-Guided Symbolic Regression under Extrapolation](https://arxiv.org/abs/2610.42024).
- Source: [MostafaCham/LLM-SR-UQ](https://github.com/MostafaCham/LLM-SR-UQ).
- Vendored commit: `b58630470144a3da6944c8f45bdbef63af38d460`.
- All 25 files in `upstream/` are byte-for-byte copies at that commit.
- Original license: MIT, copyright (c) 2026 Anonymous Authors; preserved in [upstream/LICENSE](upstream/LICENSE).
- `source_manifest.json` records the original Git blob hashes and byte lengths.

## Data and model

Data and model weights are not distributed in this repository. Download them from their respective publishers and follow their terms.

- Data mirror used by the original loader: [pkuHaowei/llm-srbench](https://huggingface.co/datasets/pkuHaowei/llm-srbench), revision `f4101e201952bb35cab8332b6dcf337b6df02aa8`. Expected hashes are in `data_manifest.json`. The downloader also retrieves the dataset README.
- Model: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), under the publisher's Llama 3.1 license. Access may require agreement through Hugging Face.
- Optional quantized backend: [Ollama llama3.1:8b-instruct-q4_K_M](https://ollama.com/library/llama3.1:8b-instruct-q4_K_M). Its results must be labeled as quantized-backend experiments.
