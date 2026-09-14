import os
import json
import re
import pandas as pd

# ============================================================
# PATHS
# ============================================================

IMAGE_ROOT = r"C:\PlantVillage-GitHub\raw\color"
MAP_FILE = r"C:\PlantVillage-GitHub\leaf_grouping\leaf-map.json"

OUTPUT_DIR = r"C:\Plant-Disease-Detection\data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "plantvillage_leaf_mapping.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_name(text):
    """
    Convert filenames/map keys to a comparable format.
    """

    text = str(text).lower().strip()

    # Remove extension
    text = re.sub(r"\.(jpg|jpeg|png)$", "", text)

    # If dataset filename contains UUID___PREFIX,
    # keep only the part after ___
    if "___" in text:
        text = text.split("___")[-1]

    # Remove copy / copy 2 / copy 3 etc.
    text = re.sub(r"\s+copy(?:\s+\d+)?$", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD MAP
# ============================================================

print("Loading leaf-map.json...")

with open(MAP_FILE, "r", encoding="utf-8") as f:
    leaf_map = json.load(f)

print("Leaf-map entries:", len(leaf_map))


# ============================================================
# CREATE NORMALIZED MAP
# ============================================================

print("Creating normalized lookup...")

normalized_map = {}

for key, values in leaf_map.items():

    normalized_key = normalize_name(key)

    if normalized_key not in normalized_map:
        normalized_map[normalized_key] = []

    normalized_map[normalized_key].extend(values)

print("Normalized map keys:", len(normalized_map))


# ============================================================
# SCAN DATASET
# ============================================================

print()
print("Scanning dataset...")
print()

rows = []

total_images = 0
matched = 0
unmatched = 0
multiple = 0


for class_name in sorted(os.listdir(IMAGE_ROOT)):

    class_path = os.path.join(IMAGE_ROOT, class_name)

    if not os.path.isdir(class_path):
        continue

    print("Processing:", class_name)

    # os.listdir is enough here.
    # No unnecessary os.path.isfile() call.

    for filename in os.listdir(class_path):

        # Only process image files
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        total_images += 1

        normalized_filename = normalize_name(filename)

        candidates = normalized_map.get(
            normalized_filename,
            []
        )

        leaf_id = ""
        status = "unmatched"

        if candidates:

            # Keep only candidates belonging to this class
            class_candidates = [
                c for c in candidates
                if str(c).startswith(class_name + ":::")
            ]

            if len(class_candidates) == 1:

                leaf_id = class_candidates[0]
                status = "matched"
                matched += 1

            elif len(class_candidates) > 1:

                leaf_id = class_candidates[0]
                status = "multiple_candidates"
                multiple += 1
                matched += 1

            elif len(candidates) == 1:

                # Candidate exists but class name did not match
                leaf_id = candidates[0]
                status = "matched_class_mismatch"
                matched += 1

            else:

                status = "multiple_candidates"

                leaf_id = candidates[0]

                multiple += 1
                matched += 1

        else:

            unmatched += 1

        rows.append({
            "image_path": os.path.join(
                class_path,
                filename
            ),
            "filename": filename,
            "class": class_name,
            "leaf_id": leaf_id,
            "match_status": status
        })


# ============================================================
# SAVE CSV
# ============================================================

print()
print("Creating DataFrame...")

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("LEAF MATCHING COMPLETE")
print("=" * 60)

print("Total images:", total_images)
print("Matched:", matched)
print("Unmatched:", unmatched)
print("Multiple candidates:", multiple)

if total_images:
    print(
        "Match percentage:",
        round(
            matched / total_images * 100,
            2
        ),
        "%"
    )

print()
print("Status distribution:")
print(df["match_status"].value_counts())

print()
print("CSV saved:")
print(OUTPUT_FILE)

print()
print("First 20 unmatched images:")

unmatched_df = df[
    df["match_status"] == "unmatched"
]

for _, row in unmatched_df.head(20).iterrows():

    print(
        row["filename"],
        "->",
        row["class"]
    )

print()
print("DONE.")