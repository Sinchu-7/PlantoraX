import os
import pandas as pd

IMAGE_ROOT = r"C:\PlantVillage-GitHub\raw\color"
MAP_ROOT = r"C:\PlantVillage-GitHub\leaf_grouping\filtered_leafmaps"

classes = [
    d for d in os.listdir(IMAGE_ROOT)
    if os.path.isdir(os.path.join(IMAGE_ROOT, d))
]

print("Number of image classes:", len(classes))
print()

for class_name in classes:
    map_file = os.path.join(MAP_ROOT, class_name + ".csv")

    if not os.path.exists(map_file):
        print("NO MAP:", class_name)
        continue

    df = pd.read_csv(map_file)

    print(
        f"{class_name}: "
        f"{len(df)} mapped images, "
        f"{df['Leaf #'].nunique()} leaf groups"
    )