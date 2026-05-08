"""
Generate model responses for paired patient profiles using a HuggingFace causal LM.

Input CSV requirements:
- Must contain columns:
    * prompt_original
    * prompt_location

Output CSV:
- Columns:
    * pair_id
    * prompt_type
    * prompt
    * response

Dependencies:
pip install transformers torch pandas sentencepiece protobuf huggingface_hub
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
model_name = "your-chosen-model"
input_csv = "input_patient_profiles.csv"
output_csv = "model_responses.csv"

max_new_tokens = 1000

required_columns = [
    "prompt_original",
    "prompt_location"
]

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype
).to(device)

model.eval()

# Helpers
def normalize_bullets(text: str) -> str:
    bullet_chars = ['‚Ä¢', '\u2022', '*', '-', '‣', '⁃']
    for b in bullet_chars:
        text = text.replace(b, '•')
    return text


def generate_response(prompt: str, max_new_tokens: int = max_new_tokens) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return normalize_bullets(response)

def main():

    df = pd.read_csv(input_csv, encoding="utf-8")

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain column '{col}'")

    results = []

    for pair_id, row in enumerate(df.itertuples(index=False)):
        print(f"Processing pair {pair_id}...")

        for prompt_col in required_columns:

            prompt = getattr(row, prompt_col)

            try:
                response = generate_response(prompt)
            except Exception as e:
                print(f"Error on pair {pair_id}, {prompt_col}: {e}")
                response = ""

            results.append({
                "pair_id": pair_id,
                "prompt_type": prompt_col,
                "prompt": prompt,
                "response": response
            })

    pd.DataFrame(results).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved results to: {output_csv}")


if __name__ == "__main__":
    main()