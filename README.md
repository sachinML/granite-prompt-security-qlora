# Multilingual Prompt-Attack Classification with Granite 3.3 2B and QLoRA

Parameter-efficient fine-tuning of **IBM Granite 3.3 2B Instruct** to classify multilingual text as **BENIGN**, **JAILBREAK**, or **PROMPT_INJECTION**. This project uses the [RMIT Hackathon 2026 Kaggle competition](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview) dataset and compares the pretrained model with its QLoRA-adapted version on the **same 2,800 held-out validation examples**.

## Results

| Metric | Base Granite | QLoRA fine-tuned | Change |
|---|---:|---:|---:|
| Accuracy | 0.6482 | **0.9054** | +25.72 percentage points |
| Macro-F1 | 0.4578 | **0.8560** | +0.3982 |

**Fine-tuned per-class validation metrics** (rounded from the classification report):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BENIGN | 0.95 | 0.90 | 0.93 | 1,736 |
| JAILBREAK | 0.84 | 0.96 | 0.90 | 868 |
| PROMPT_INJECTION | 0.85 | 0.66 | 0.74 | 196 |

These are **local validation results**, not a Kaggle leaderboard score or an independent external test result. Both models were evaluated on the same validation examples using the same three-label likelihood-scoring procedure. The pretrained model was evaluated with LoRA adapters temporarily disabled.

## Problem

Security classifiers must distinguish ordinary user requests from attempts to bypass safety restrictions or redirect an AI assistant's instructions. This dataset contains prompts in **Portuguese (`pt`), French (`fr`), Italian (`it`) and English (`en`)**. The minority `PROMPT_INJECTION` class is particularly important to evaluate separately: a model with good overall accuracy can still miss many examples in this class.

We investigate whether a small, quantized language model can learn this task with a lightweight LoRA adapter instead of full-parameter fine-tuning.

## Data and splits

| Item | Count |
|---|---:|
| Labeled competition training set | 14,000 |
| Unlabeled competition test set | 6,000 |
| Internal training split (80%) | 11,200 |
| Training after excluding >1,024-token formatted sequences | **11,083** |
| Internal held-out validation split (20%) | **2,800** |

The 80/20 split used `random_state=42` with stratification on the combined **language + label** field. There were no exact duplicate `text` values in the original 14,000 training examples, although this does not rule out paraphrase or cross-language leakage.

Class distribution in the labeled data: **62% BENIGN, 31% JAILBREAK, 7% PROMPT_INJECTION**. The full chat-formatted sequence includes system instructions, the user text and the assistant label. We filtered **117** training examples whose full formatted sequence exceeded 1,024 tokens, rather than risk truncating the label at the end of the sequence.

Dataset source: [RMIT Hackathon 2026 on Kaggle](https://www.kaggle.com/competitions/rmit-hackathon-2026/data). Competition datasets are **not redistributed in this repository**; obtain them through Kaggle, subject to the competition's access and use terms.

## Method

### Baseline

The pretrained Granite model was loaded with 4-bit quantization and instructed to return one of three class labels. Initial free-form generation sometimes produced explanations or followed instructions embedded in the input, leading to malformed labels. We therefore used **candidate-label likelihood scoring**:

1. Build the same chat prompt for every example.
2. For each valid class label, calculate the sum of causal token log probabilities for its complete token sequence.
3. Select the highest-scoring label.

This guarantees an in-vocabulary class choice, **not** correct classification. Using summed log probabilities may also favor shorter token sequences; the method was held constant between the baseline and adapted model for the matched comparison. The output of the classifier is not a calibrated risk probability.

### QLoRA training

We used conversational prompt-completion examples and **completion-only supervised fine-tuning**. The model receives the whole prompt as context, but the training loss is computed only on the assistant completion (the correct class label and associated end-of-message token). The original quantized base weights remain frozen while LoRA parameters are updated.

| Configuration | Value |
|---|---|
| Base model | Granite 3.3 2B Instruct |
| Fine-tuning approach | 4-bit QLoRA + completion-only SFT |
| Quantization | NF4, double quantization |
| Mixed-precision compute | FP16 |
| LoRA rank / alpha / dropout | `8` / `16` / `0.05` |
| Target modules | `all-linear` |
| Trainable parameters | **14,090,240** (~0.5531% of total) |
| Maximum formatted sequence length | `1024` |
| Per-device batch size | `1` |
| Gradient accumulation | `8` |
| Effective batch size | `8` (single GPU) |
| Learning rate | `2e-4` |
| Scheduler / warmup | Linear / 6 steps |
| Optimizer | `adamw_torch` |
| Main training duration | **200 optimizer steps** (~1,600 example presentations) |
| Hardware | Kaggle NVIDIA Tesla T4 (single visible GPU) |
| Mean training loss over main run | **0.1109** |

The Kaggle environment exposed two Tesla T4 GPUs, but training was configured to use **only one** to avoid unsupported `nn.DataParallel` handling in the trainer. Another environment issue required casting the trainable LoRA adapter weights to **FP32 after trainer initialization** to avoid a BF16-gradient/FP16-scaler mismatch. The 4-bit base model weights remained quantized.

## Evaluation and error analysis

A smaller, representative 500-example validation subsample initially yielded **0.8799 macro-F1**. Evaluating **all 2,800 validation examples** yielded **0.8560 macro-F1**, so the full validation metric is the main reported result.

The biggest remaining challenge is `PROMPT_INJECTION` recall (**0.66**). In the 500-example multilingual analysis, several Portuguese benign examples involving fictional scenarios and role-play were incorrectly classified as jailbreaks. A small counterfactual check of simple role-play prompts did **not** reproduce that behavior, so we do **not** claim that role-play wording alone caused those errors. Some dataset examples also have ambiguous boundaries between jailbreaks and prompt injections.

The full validation set has been inspected repeatedly during analysis. Further tuning against these observations would turn it into a development set. An untouched external or official hidden test set would be needed for an unbiased final estimate.

## Repository contents

Suggested layout (adapt paths to the files in your local repository):

```text
granite-prompt-security-qlora/
├── notebooks/
│   └── granite_qlora_training.ipynb
├── results/
│   ├── evaluation_metrics.json
│   ├── full_evaluation_metrics.json
│   └── [optional aggregate analysis]
├── .gitignore
└── README.md
```

The local backup may additionally include `adapter_model.safetensors`, `adapter_config.json`, tokenizer files and row-level prediction CSVs. **Do not commit large weights, raw competition data or row-level competition predictions without checking repository size and redistribution rules.** Store the adapter in a private Kaggle Dataset or other appropriate model artifact store.

### Running the experiment

1. Open the notebook in **Kaggle**, attach the competition data and approved Granite model, and enable a T4 GPU.
2. Set `CUDA_VISIBLE_DEVICES=0` **before importing PyTorch**.
3. Load the competition `train.csv`, recreate the language-and-label stratified split with seed 42, and format prompt-completion examples.
4. Quantize the base model in 4-bit NF4; configure LoRA and the SFT trainer with completion-only loss.
5. Check that only the completion tokens are supervised and that trainable adapter parameters are FP32 in the T4 mixed-precision setup.
6. Train the adapter, save its weights and configuration, and evaluate the adapter-enabled and adapter-disabled model with the **same** scoring function and validation records.

**Notebook status:** The original notebook is an exploratory training record and may contain repeated cells from debugging; it should be cleaned before promising one-click reproducibility. Exact dependency versions should be pinned from the working Kaggle environment.

## Limitations

- Results come from **one Kaggle competition dataset** and do not establish real-world attack-detection effectiveness.
- The dataset's `BENIGN`, `JAILBREAK` and `PROMPT_INJECTION` categories can be semantically ambiguous for certain examples.
- Full validation was reused for analysis; it is not an untouched final test after iterative inspection.
- The label-likelihood decision rule is fixed across both models but is not calibrated and may be sensitive to class-label tokenization/length.
- The 200-step experiment did not cover a full epoch, and no broad hyperparameter sweep or multi-seed confidence interval was performed.
- **No official Kaggle leaderboard score is claimed.**

## Tools

Python · PyTorch · Hugging Face Transformers · PEFT · TRL · bitsandbytes · scikit-learn · pandas · Kaggle GPU

## Acknowledgment

Dataset and task: [RMIT Hackathon 2026 — Kaggle](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview). Base model: [IBM Granite](https://huggingface.co/ibm-granite).
