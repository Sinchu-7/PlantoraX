import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Plant-Disease-Detection"

MAPPING_FILE = os.path.join(
    BASE_DIR,
    "data",
    "plantvillage_leaf_mapping.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "splits"
)

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
VAL_FILE = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET MAPPING
# ============================================================

print("Loading dataset mapping...")

if not os.path.exists(MAPPING_FILE):
    raise FileNotFoundError(
        f"\nMapping file not found:\n{MAPPING_FILE}"
    )

df = pd.read_csv(MAPPING_FILE)

print(f"Total images: {len(df)}")

print("\nCSV columns found:")
print(list(df.columns))


# ============================================================
# FIX COLUMN NAME
# ============================================================

# Your CSV contains "class", while our script uses "class_name".

if "class" in df.columns and "class_name" not in df.columns:

    df = df.rename(
        columns={
            "class": "class_name"
        }
    )

    print("\n✓ Renamed 'class' column to 'class_name'.")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_path",
    "class_name",
    "leaf_id"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nRequired column '{column}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )


# ============================================================
# CLEAN DATA
# ============================================================

df["image_path"] = df["image_path"].astype(str)
df["class_name"] = df["class_name"].astype(str)

df["leaf_id"] = (
    df["leaf_id"]
    .fillna("")
    .astype(str)
)


# ============================================================
# DATASET INFORMATION
# ============================================================

classes = sorted(
    df["class_name"].unique()
)

print(
    f"\nTotal classes: {len(classes)}"
)

print("\nClasses found:")

for class_name in classes:
    print(f"  {class_name}")


# ============================================================
# CHECK IMAGE PATHS
# ============================================================

print("\nChecking image paths...")

valid_paths = []
invalid_paths = []

for path in df["image_path"]:

    if os.path.isabs(path):

        full_path = path

    else:

        full_path = os.path.join(
            BASE_DIR,
            path
        )

    if os.path.exists(full_path):

        valid_paths.append(True)

    else:

        valid_paths.append(False)
        invalid_paths.append(path)


df["valid_image"] = valid_paths


if invalid_paths:

    print(
        f"WARNING: {len(invalid_paths)} image paths "
        f"could not be found."
    )

    df = df[
        df["valid_image"] == True
    ].copy()

else:

    print(
        "✓ All image paths are valid."
    )


# ============================================================
# IDENTIFY MAPPED / UNMATCHED IMAGES
# ============================================================

mapped_df = df[
    (df["leaf_id"].notna()) &
    (df["leaf_id"].str.strip() != "")
].copy()


unmatched_df = df[
    (df["leaf_id"].isna()) |
    (df["leaf_id"].str.strip() == "")
].copy()


print(
    f"\nMapped images: {len(mapped_df)}"
)

print(
    f"Unmatched images: {len(unmatched_df)}"
)


# ============================================================
# CREATE GROUP ID
# ============================================================

if len(mapped_df) > 0:

    mapped_df["group_id"] = (
        mapped_df["class_name"].astype(str)
        + "|||"
        + mapped_df["leaf_id"].astype(str)
    )

else:

    mapped_df["group_id"] = ""


# ============================================================
# LEAKAGE-AWARE SPLIT
# ============================================================

print(
    "\nCreating leakage-aware split for mapped images..."
)


mapped_train_list = []
mapped_val_list = []
mapped_test_list = []


for class_name in classes:

    class_data = mapped_df[
        mapped_df["class_name"] == class_name
    ].copy()


    if len(class_data) == 0:

        continue


    groups = class_data[
        "group_id"
    ].unique()


    num_groups = len(groups)


    print(
        f"\n{class_name}: "
        f"{len(class_data)} images, "
        f"{num_groups} leaf groups"
    )


    # --------------------------------------------------------
    # TOO FEW GROUPS
    # --------------------------------------------------------

    if num_groups < 3:

        print(
            "  WARNING: Fewer than 3 leaf groups."
        )

        print(
            "  Putting all mapped images into TRAIN."
        )

        mapped_train_list.append(
            class_data
        )

        continue


    # --------------------------------------------------------
    # 70% TRAIN / 30% TEMP
    # --------------------------------------------------------

    train_groups, temp_groups = train_test_split(

        groups,

        test_size=0.30,

        random_state=RANDOM_STATE
    )


    # --------------------------------------------------------
    # 15% VALIDATION / 15% TEST
    # --------------------------------------------------------

    val_groups, test_groups = train_test_split(

        temp_groups,

        test_size=0.50,

        random_state=RANDOM_STATE
    )


    train_data = class_data[
        class_data["group_id"].isin(
            train_groups
        )
    ].copy()


    val_data = class_data[
        class_data["group_id"].isin(
            val_groups
        )
    ].copy()


    test_data = class_data[
        class_data["group_id"].isin(
            test_groups
        )
    ].copy()


    mapped_train_list.append(
        train_data
    )

    mapped_val_list.append(
        val_data
    )

    mapped_test_list.append(
        test_data
    )


# ============================================================
# COMBINE MAPPED DATA
# ============================================================

if mapped_train_list:

    mapped_train = pd.concat(
        mapped_train_list,
        ignore_index=True
    )

else:

    mapped_train = pd.DataFrame(
        columns=mapped_df.columns
    )


if mapped_val_list:

    mapped_val = pd.concat(
        mapped_val_list,
        ignore_index=True
    )

else:

    mapped_val = pd.DataFrame(
        columns=mapped_df.columns
    )


if mapped_test_list:

    mapped_test = pd.concat(
        mapped_test_list,
        ignore_index=True
    )

else:

    mapped_test = pd.DataFrame(
        columns=mapped_df.columns
    )


# ============================================================
# SPLIT UNMATCHED IMAGES
# ============================================================

print(
    "\nSplitting unmatched images..."
)


if len(unmatched_df) > 0:

    class_counts = (
        unmatched_df[
            "class_name"
        ].value_counts()
    )


    minimum_class_count = (
        class_counts.min()
    )


    print(
        f"Minimum unmatched images "
        f"in any class: "
        f"{minimum_class_count}"
    )


    # ========================================================
    # STRATIFIED SPLIT
    # ========================================================

    if minimum_class_count >= 2:

        print(
            "All unmatched classes have "
            "at least 2 images."
        )

        print(
            "Using stratified splitting..."
        )


        unmatched_train, unmatched_temp = (
            train_test_split(

                unmatched_df,

                test_size=0.30,

                random_state=RANDOM_STATE,

                stratify=unmatched_df[
                    "class_name"
                ]
            )
        )


        temp_counts = (
            unmatched_temp[
                "class_name"
            ].value_counts()
        )


        # ====================================================
        # VALIDATION + TEST
        # ====================================================

        if temp_counts.min() >= 2:

            unmatched_val, unmatched_test = (
                train_test_split(

                    unmatched_temp,

                    test_size=0.50,

                    random_state=RANDOM_STATE,

                    stratify=unmatched_temp[
                        "class_name"
                    ]
                )
            )

        else:

            print(
                "Temporary set contains classes "
                "with fewer than 2 images."
            )

            print(
                "Using random split for "
                "validation/test."
            )


            unmatched_val, unmatched_test = (
                train_test_split(

                    unmatched_temp,

                    test_size=0.50,

                    random_state=RANDOM_STATE,

                    shuffle=True
                )
            )


    # ========================================================
    # RANDOM SPLIT
    # ========================================================

    else:

        print(
            "Some unmatched classes have "
            "fewer than 2 images."
        )

        print(
            "Using random splitting..."
        )


        unmatched_train, unmatched_temp = (
            train_test_split(

                unmatched_df,

                test_size=0.30,

                random_state=RANDOM_STATE,

                shuffle=True
            )
        )


        unmatched_val, unmatched_test = (
            train_test_split(

                unmatched_temp,

                test_size=0.50,

                random_state=RANDOM_STATE,

                shuffle=True
            )
        )


else:

    unmatched_train = pd.DataFrame(
        columns=unmatched_df.columns
    )

    unmatched_val = pd.DataFrame(
        columns=unmatched_df.columns
    )

    unmatched_test = pd.DataFrame(
        columns=unmatched_df.columns
    )


print(
    f"Unmatched train: "
    f"{len(unmatched_train)}"
)

print(
    f"Unmatched validation: "
    f"{len(unmatched_val)}"
)

print(
    f"Unmatched test: "
    f"{len(unmatched_test)}"
)


# ============================================================
# COMBINE MAPPED + UNMATCHED
# ============================================================

train_df = pd.concat(
    [
        mapped_train,
        unmatched_train
    ],
    ignore_index=True
)


validation_df = pd.concat(
    [
        mapped_val,
        unmatched_val
    ],
    ignore_index=True
)


test_df = pd.concat(
    [
        mapped_test,
        unmatched_test
    ],
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


validation_df = validation_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# REMOVE INTERNAL COLUMN
# ============================================================

for dataframe in [
    train_df,
    validation_df,
    test_df
]:

    if "valid_image" in dataframe.columns:

        dataframe.drop(
            columns=["valid_image"],
            inplace=True
        )


# ============================================================
# SAVE CSV FILES
# ============================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

validation_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# FINAL DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET SPLIT COMPLETED")
print("=" * 70)


print(
    f"\nTrain:       {len(train_df)}"
)

print(
    f"Validation:  {len(validation_df)}"
)

print(
    f"Test:        {len(test_df)}"
)


total_split = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)


print(
    f"Total:       {total_split}"
)


print(
    f"\nOriginal valid images: "
    f"{len(df)}"
)


if total_split == len(df):

    print(
        "✓ All valid images are included."
    )

else:

    print(
        "⚠ WARNING: Split total does not "
        "match valid image count."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


train_counts = (
    train_df[
        "class_name"
    ].value_counts()
)


val_counts = (
    validation_df[
        "class_name"
    ].value_counts()
)


test_counts = (
    test_df[
        "class_name"
    ].value_counts()
)


distribution = pd.DataFrame({

    "Train": train_counts,

    "Validation": val_counts,

    "Test": test_counts

}).fillna(0).astype(int)


distribution = distribution.sort_index()


print("\n")
print(
    distribution.to_string()
)


# ============================================================
# CHECK ALL CLASSES
# ============================================================

train_classes = set(
    train_df[
        "class_name"
    ].unique()
)


val_classes = set(
    validation_df[
        "class_name"
    ].unique()
)


test_classes = set(
    test_df[
        "class_name"
    ].unique()
)


print("\n")
print("=" * 70)
print("CLASS PRESENCE CHECK")
print("=" * 70)


print(
    f"\nClasses in train:       "
    f"{len(train_classes)}"
)


print(
    f"Classes in validation:  "
    f"{len(val_classes)}"
)


print(
    f"Classes in test:        "
    f"{len(test_classes)}"
)


missing_train = (
    set(classes) - train_classes
)


missing_val = (
    set(classes) - val_classes
)


missing_test = (
    set(classes) - test_classes
)


if missing_train:

    print(
        "\n⚠ Missing from TRAIN:"
    )

    for c in sorted(missing_train):

        print(
            f"  {c}"
        )

else:

    print(
        "\n✓ All classes are present in TRAIN."
    )


if missing_val:

    print(
        "\n⚠ Missing from VALIDATION:"
    )

    for c in sorted(missing_val):

        print(
            f"  {c}"
        )

else:

    print(
        "✓ All classes are present in VALIDATION."
    )


if missing_test:

    print(
        "\n⚠ Missing from TEST:"
    )

    for c in sorted(missing_test):

        print(
            f"  {c}"
        )

else:

    print(
        "✓ All classes are present in TEST."
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n")
print("=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)


def get_groups(data):

    if "group_id" not in data.columns:

        return set()


    groups = (
        data[
            "group_id"
        ]
        .dropna()
        .astype(str)
    )


    groups = groups[
        groups.str.strip() != ""
    ]


    return set(groups)


train_groups = get_groups(
    train_df
)

val_groups = get_groups(
    validation_df
)

test_groups = get_groups(
    test_df
)


train_val_overlap = (
    train_groups.intersection(
        val_groups
    )
)


train_test_overlap = (
    train_groups.intersection(
        test_groups
    )
)


val_test_overlap = (
    val_groups.intersection(
        test_groups
    )
)


print(
    f"\nTrain ↔ Validation overlap: "
    f"{len(train_val_overlap)}"
)


print(
    f"Train ↔ Test overlap: "
    f"{len(train_test_overlap)}"
)


print(
    f"Validation ↔ Test overlap: "
    f"{len(val_test_overlap)}"
)


if (

    len(train_val_overlap) == 0

    and

    len(train_test_overlap) == 0

    and

    len(val_test_overlap) == 0

):

    print(
        "\n✓ NO LEAF GROUP LEAKAGE DETECTED."
    )

else:

    print(
        "\n⚠ WARNING: Leaf groups overlap."
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_file = os.path.join(
    OUTPUT_DIR,
    "class_names.txt"
)


with open(
    class_names_file,
    "w",
    encoding="utf-8"
) as f:

    for class_name in classes:

        f.write(
            class_name + "\n"
        )


# ============================================================
# FINAL FILE LOCATIONS
# ============================================================

print("\n")
print("=" * 70)
print("FILES CREATED")
print("=" * 70)


print(
    f"\nTrain CSV:\n{TRAIN_FILE}"
)


print(
    f"\nValidation CSV:\n{VAL_FILE}"
)


print(
    f"\nTest CSV:\n{TEST_FILE}"
)


print(
    f"\nClass names:\n{class_names_file}"
)


print("\n")
print("=" * 70)
print("READY FOR MODEL TRAINING")
print("=" * 70)