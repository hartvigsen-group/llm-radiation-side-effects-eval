"""
This script evaluates model-generated side effect predictions for breast cancer radiation patients. It calculates precision, recall, and F1 against ground truth side effects for location-specific prompts, and computes pairwise overlap between response sets (e.g., original vs. location-specific).

Expected Inputs:
1. Model response file(s):
   - Sheet with location-specific prompts (used for precision, recall, F1 metrics):
       Required columns: 'prompt', 'response', 'radiation_added'
   - Sheet with all prompts (used for pairwise overlap between responses):
       Required columns: 'pair_id', 'prompt', 'prompt_type', 'response'

2. Ground truth Excel file listing radiation site side effects:
   - 'Side effect' column
   - One column per radiation site indicating presence (e.g., 'x')

Outputs:
- CSV of side effect metrics per response (precision, recall, F1), e.g., side_effect_metrics.csv
- CSV of pairwise overlap between response sets, e.g., overlap_metrics.csv
"""

import pandas as pd
import json
from tqdm import tqdm
from openai import AzureOpenAI
import os


# Azure OpenAI setup 
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = "o4-mini"
API_VERSION = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=API_VERSION,
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
)

# File paths/sheet names
model_responses_file = "responses.xls"
ground_truth_file = "breast-rt-side-effects.xlsx"
location_sheet = "Location"
all_prompts_sheet = "responses"

# Output files
metrics_output_file = "metrics.csv"
overlap_output_file = "overlap.csv"

# Extract bullet list side effects
def extract_side_effects(text):
    if pd.isna(text):
        return []
    bullets = text.split("•")
    return [b.strip().lower() for b in bullets if b.strip()]

# Load model responses
responses_df = pd.read_excel(model_responses_file, sheet_name=location_sheet)
responses_df["predicted_effects"] = responses_df["response"].apply(extract_side_effects)

# Load ground truth side effects
truth_df = pd.read_excel(ground_truth_file).fillna("")
radiation_site_columns = truth_df.columns[2:]  

radiation_to_side_effects = {}
for site in radiation_site_columns:
    side_effects = truth_df[truth_df[site] == "x"]["Side effect"]
    cleaned = side_effects.str.lower().str.strip()
    radiation_to_side_effects[site.lower().strip()] = set(cleaned)

# Semantic matching
def gpt_check_batch(predicted_list, ground_truth):
    if not predicted_list or not ground_truth:
        return set()
    
    prompt = f"""
Determine which of the following predicted side effects are present in the ground truth list.
Include side effects that match in meaning, even if wording differs.
Answer only with a comma-separated list of side effects that match.

Predicted: {predicted_list}
Ground truth: {ground_truth}
"""
    completion = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    response_json = json.loads(completion.model_dump_json())
    matched_text = response_json["choices"][0]["message"]["content"].strip().lower()
    return set([x.strip() for x in matched_text.split(",") if x.strip()])

# Compute Precision, Recall, F1
def compute_metrics_gpt(row):
    radiation_site = str(row["radiation_added"]).lower().strip()
    predicted = row["predicted_effects"]
    ground_truth = list(radiation_to_side_effects.get(radiation_site, []))

    if not ground_truth:
        return pd.Series({
            "ground_truth_effects": [],
            "correct_effects": [],
            "precision": None,
            "recall": None,
            "f1": None,
            "model_name": DEPLOYMENT_NAME
        })

    correct_set = gpt_check_batch(predicted, ground_truth)

    precision = len(correct_set) / len(predicted) if predicted else 0
    recall = len(correct_set) / len(ground_truth) if ground_truth else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return pd.Series({
        "ground_truth_effects": ", ".join(ground_truth),
        "correct_effects": ", ".join(correct_set),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "model_name": DEPLOYMENT_NAME
    })

# Compute metrics for all location-specific responses
metrics_list = []
for index in tqdm(range(len(responses_df)), desc="Computing metrics"):
    row = responses_df.iloc[index]
    metrics_list.append(compute_metrics_gpt(row))

metrics_df = pd.DataFrame(metrics_list)
final_df = pd.concat([responses_df, metrics_df], axis=1)
final_df.to_csv(metrics_output_file, index=False)

# Compute overlap between prompt_original and prompt_location
df_overlap_input = pd.read_excel(model_responses_file, sheet_name=all_prompts_sheet)

pairwise_overlap = []
for pair_id, group in df_overlap_input.groupby("pair_id"):
    original_row = group[group["prompt_type"] == "prompt_original"]
    location_row = group[group["prompt_type"] == "prompt_location"]

    if original_row.empty or location_row.empty:
        continue

    original_response = extract_side_effects(original_row.iloc[0]["response"])
    location_response = extract_side_effects(location_row.iloc[0]["response"])

    # Semantic matches using GPT
    matched_set = gpt_check_batch(original_response, location_response)

    # Limit to terms actually mentioned in either response
    all_terms = {t.strip().lower() for t in (original_response + location_response)}
    matched_set = matched_set.intersection(all_terms)

    union_count = len(all_terms)
    overlap_ratio = len(matched_set) / union_count if union_count else 0

    pairwise_overlap.append({
        "pair_id": pair_id,
        "original_effects": ", ".join(original_response),
        "location_effects": ", ".join(location_response),
        "overlapping_effects": ", ".join(matched_set),
        "original_count": len(original_response),
        "location_count": len(location_response),
        "overlap_count": len(matched_set),
        "overlap_ratio": overlap_ratio
    })

overlap_df = pd.DataFrame(pairwise_overlap)
overlap_df.to_csv(overlap_output_file, index=False)
