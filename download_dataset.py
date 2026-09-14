import tensorflow_datasets as tfds

print("Starting PlantVillage download...")

dataset, info = tfds.load(
    "plant_village",
    with_info=True,
    as_supervised=True,
    data_dir=r"C:\PlantVillage"
)

print("\nDownload complete!")
print("Number of classes:", info.features["label"].num_classes)
print("Classes:")

for i, name in enumerate(info.features["label"].names):
    print(i, name)