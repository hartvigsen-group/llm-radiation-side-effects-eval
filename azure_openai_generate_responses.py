"""
Generate model responses for paired patient profiles using Azure OpenAI.

Input CSV requirements:
- File should have columns:
    * prompt_original : original patient profile prompt
    * prompt_location : patient profile with radiation location specified

Output CSV:
- Columns:
    * pair_id           : integer ID for each profile pair
    * prompt_type       : 'prompt_original' or 'prompt_location'
    * prompt            : the input prompt
    * response          : model-generated response (bullet points normalized to "•")
"""

import pandas as pd
import json
import os
from openai import AzureOpenAI

input_file = "data/patient_prompt_pairs.csv"
output_file = "results/model_responses.csv"

# Azure OpenAI configuration
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = "o4-mini"
API_VERSION = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=API_VERSION,
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
)

def generate_response(prompt):
    completion = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    response_json = json.loads(completion.model_dump_json())

    return response_json["choices"][0]["message"]["content"].strip()


def main():

    # Load dataset
    df = pd.read_csv(
        input_file,
        encoding="macroman"
    )

    results = []

    # Generate responses
    for pair_id, row in enumerate(df.itertuples(index=False)):

        for prompt_col in ["prompt_original", "prompt_location"]:

            prompt = getattr(row, prompt_col)

            response = generate_response(prompt)

            # Normalize bullet characters
            for bullet_char in ['‚Ä¢', '\u2022', '*', '-', '‣', '⁃']:
                response = response.replace(bullet_char, '•')

            response = response.strip()

            results.append({
                "prompt_type": prompt_col,
                "prompt": prompt,
                "response": response,
                "pair_id": pair_id
            })

    # Save results
    pd.DataFrame(results).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()