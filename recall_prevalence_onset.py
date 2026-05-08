"""
This script computes recall of model-predicted side effects stratified by temporal onset and side effect frequency.

Expected Inputs:
1. Model response file (Excel):
   - Sheet with location-specific prompts
   - Required columns: 'prompt', 'response', 'radiation_added'

2. Ground truth Excel file:
   - Column 'Side effect' for the side effect term
   - Column 'term' for occurrence/commonness category (e.g., short-term common)
   - One column per radiation site indicating presence ('x')

Expected Outputs:
- CSV of recall per radiation site occurrence category
"""

import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI
import json
import os

# Azure OpenAI setup
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = "<YOUR_MODEL_DEPLOYMENT>"
API_VERSION = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=API_VERSION,
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
)

# Load model responses
MODEL_RESPONSES_FILE = "<MODEL_RESPONSES_FILE>.xlsx"
LOCATION_SHEET = "<LOCATION_SHEET>"  # sheet with location-specific prompts

responses_df = pd.read_excel(MODEL_RESPONSES_FILE, sheet_name=LOCATION_SHEET)

output_file = "recall_by_commonness.csv"

# Extract bullet list side effects
def extract_side_effects(text):
    if pd.isna(text):
        return []
    bullets = text.split("•")
    return [b.strip().lower() for b in bullets if b.strip()]

responses_df["predicted_effects"] = responses_df["response"].apply(extract_side_effects)

# Load ground truth
GROUND_TRUTH_FILE = "<GROUND_TRUTH_FILE>.xlsx"
truth_df = pd.read_excel(GROUND_TRUTH_FILE).fillna("")
truth_df.rename(columns={truth_df.columns[0]: 'term'}, inplace=True)

radiation_site_columns = [col for col in truth_df.columns if col not in ['term', 'Side effect']]

# Create dictionary: radiation site → occurrence category → side effects
radiation_to_side_effects_by_commonness = {}
for site in radiation_site_columns:
    nested_dict = {}
    for _, row in truth_df.iterrows():
        if row[site] == 'x':
            commonness = row['term'].lower().strip()
            side_effect = row['Side effect'].lower().strip()
            nested_dict.setdefault(commonness, []).append(side_effect)
    radiation_to_side_effects_by_commonness[site.lower().strip()] = nested_dict

# Semantic recall mapping
def gpt_semantic_recall_mapping(predicted_list, ground_truth_dict):
    """
    Returns JSON mapping of ground truth items → semantically matched predicted items.
    Example output:
    {
      "short-term common": {
        "fatigue": ["fatigue"],
        "nipple pain": ["pain"]
      },
      ...
    }
    """
    if not predicted_list or not ground_truth_dict:
        return {k: {} for k in ground_truth_dict.keys()}

    prompt = f"""
Compare the predicted and ground truth side effects by semantic meaning.

Return JSON in this exact format:
{{
  "<commonness category>": {{
    "<ground truth item>": ["<matching predicted items>", ...],
    ...
  }},
  ...
}}

Predicted: {predicted_list}
Ground truth by commonness: {ground_truth_dict}

Only include predicted items that are semantically equivalent or highly similar in meaning to the ground truth.
"""

    completion = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        response_text = completion.choices[0].message.content.strip()
        mapping = json.loads(response_text.replace("```json", "").replace("```", ""))
    except Exception as e:
        return {k: {} for k in ground_truth_dict.keys()}

    return mapping

# Compute recall per site × occurrence category 
rows = []

def normalize_key(k):
    """Normalize key names (e.g., 'short term - rare' → 'shorttermrare')"""
    return k.lower().replace(" ", "").replace("-", "").strip()

for idx, row in tqdm(responses_df.iterrows(), total=len(responses_df)):
    radiation_site = str(row["radiation_added"]).lower().strip()
    predicted = row["predicted_effects"]
    site_ground_truth = radiation_to_side_effects_by_commonness.get(radiation_site, {})

    mapping = gpt_semantic_recall_mapping(predicted, site_ground_truth)

    # Normalize keys for consistent matching
    norm_gt = {normalize_key(k): v for k, v in site_ground_truth.items()}
    norm_mapping = {normalize_key(k): v for k, v in mapping.items()}

    for gt_key, gt_list in norm_gt.items():
        gt_mapping = norm_mapping.get(gt_key, {})

        matched_gt_count = 0
        used_preds = set()
        matched_preds = set()

        for gt_item, preds in gt_mapping.items():
            if not preds:
                continue
            new_preds = [p for p in preds if p not in used_preds]
            if new_preds:
                matched_gt_count += 1
                used_preds.update(new_preds)
                matched_preds.update(new_preds)

        recall = matched_gt_count / len(gt_list) if gt_list else 0.0

        readable_key = next((k for k in site_ground_truth if normalize_key(k) == gt_key), gt_key)

        rows.append({
            "prompt_id": row.get("prompt_id", idx),
            "radiation_added": radiation_site,
            "Occurrence": readable_key,
            "predicted_effects": predicted,
            "ground_truth_effects": gt_list,
            "correct_effects": list(matched_preds),
            "recall": recall
        })

# Save results
final_df = pd.DataFrame(rows)
final_df.to_csv(output_file, index=False)
