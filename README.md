# Can Language Models Identify Side Effects of Breast Cancer Radiation Treatments?

## Overview
This repository contains the code, data, and evaluation scripts for our 
stress-testing framework that evaluates LLM reliability for breast cancer 
radiation side-effect generation. We evaluate seven instruction-tuned LLMs 
across four prompting regimes and compare outputs to a clinician-curated 
ground truth reference.

## Data

### Patient Profiles
- **Base profiles**: 21 breast cancer patient profiles mentioning radiation 
  without specifying type or anatomical location
- **Specified profiles**: The same 21 profiles with a specific radiation 
  type/location appended (e.g., "radiation (chest wall and nodes)")
- Profiles are sourced from [AIME](https://arxiv.org/abs/2411.03395)
  and [OncQA](https://arxiv.org/abs/2310.17703), normalized into 
  a shared EHR-style format

### Ground Truth Side-Effect Reference
- Derived from informed consent forms at two major academic medical centers
- Developed by a team including more than seven breast radiation oncologists
- Each side effect is annotated by:
  - **Frequency**: Common, Uncommon, Rare, Extremely Rare
  - **Temporal onset**: Short-term, Long-term
- Covers five radiation types/locations:
  - Accelerated Partial Breast Irradiation (APBI)
  - Chest Wall
  - Chest Wall and Nodes
  - Breast and Nodes
  - Breast Only

## Setup

### 1. Clone the repository
```bash
git clone https://github.com//breast-cancer-radiation-llm.git
cd breast-cancer-radiation-llm
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables (for Azure OpenAI scripts)
```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```
## Usage

### Step 1 — Generate profiles
```bash
python code/perturb_profiles.py
```

### Step 2 — Generate model responses

For HuggingFace models (LLaMA, Gemma, Mistral, Qwen, Phi):
```bash
python code/huggingface_generate_responses.py
```
Set `model_name` at the top of the script to your chosen HuggingFace model ID.

For Azure OpenAI models (o4-mini, GPT-5):
```bash
python code/azure_openai_generate_responses.py
```

### Step 3 — Evaluate responses
```bash
python code/evaluation_metrics.py
```
Outputs:
- `results/side_effect_metrics.csv` — Precision, recall, F1 per response
- `results/overlap_metrics.csv` — Pairwise overlap between base and specified profiles

### Step 4 — Recall by frequency and temporal onset
```bash
python code/recall_prevalence_onset.py
```
Outputs `results/recall_by_commonness.csv`.


