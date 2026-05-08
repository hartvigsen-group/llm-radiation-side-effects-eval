"""
Perturb patient profiles by replacing generic mentions of 'radiation'
with specific radiation types and anatomical locations.
"""

import pandas as pd
import re

RADIATION_OPTIONS = [
    "Radiation Breast Only",
    "Radiation Breast and Nodes",
    "Radiation Chest Wall",
    "Radiation Chest Wall and Nodes",
    "Accelerated partial breast irradiation (APBI)"
]

def insert_radiation_location(text, replacement):
    return re.sub(r'\bradiation\b', replacement, text, flags=re.IGNORECASE)

def main():
    df = pd.read_csv("data/raw/ExperimentNoLocation.csv")

    perturbed_profiles = []

    for i, profile in enumerate(df["original_profile"]):
        radiation_type = RADIATION_OPTIONS[i % len(RADIATION_OPTIONS)]
        modified = insert_radiation_location(profile, radiation_type)

        perturbed_profiles.append({
            "base_profile": profile,
            "specified_profile": modified,
            "radiation_type": radiation_type
        })

    out_df = pd.DataFrame(perturbed_profiles)
    out_df.to_csv("data/processed/paired_profiles.csv", index=False)

if __name__ == "__main__":
    main()
