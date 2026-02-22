# Third-Party Notices

This project uses third-party models and datasets. This file documents attribution and licensing details for redistributed artifacts and source references.

## License Scope in This Repository

- Source code license: Apache-2.0 (see root `LICENSE`).
- Trained adapters/checkpoints and derivative model artifacts: non-commercial constrained because training used CommitBench (`Maxscha/commitbench`), which is licensed CC BY-NC 4.0 on its dataset card.

## Qwen2.5-Coder-0.5B (Base Model)

- Name: `Qwen/Qwen2.5-Coder-0.5B`
- Provider: Qwen Team (Alibaba Cloud)
- Source: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B
- License: Apache License 2.0
- License URL: https://www.apache.org/licenses/LICENSE-2.0

Notes:

- This repository primarily contains fine-tuned LoRA adapter artifacts and project code.
- Base model weights are not bundled by default in this repository.
- If you redistribute base model weights, merged checkpoints, or derived binaries, you must comply with Apache-2.0 terms, including attribution and license notice requirements.

## CommitBench Dataset

- Name: `Maxscha/commitbench`
- Source: https://huggingface.co/datasets/Maxscha/commitbench
- Dataset license: CC BY-NC 4.0 (per dataset card)
- License URL: https://creativecommons.org/licenses/by-nc/4.0/

Check the dataset card for its current license and usage constraints before redistribution or commercial use.
