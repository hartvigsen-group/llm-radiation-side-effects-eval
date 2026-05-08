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
- Profiles are sourced from [AIME](https://arxiv.org/abs/2411.03395) (18 profiles) 
  and [OncQA](https://arxiv.org/abs/2308.03853) (3 profiles), normalized into 
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

