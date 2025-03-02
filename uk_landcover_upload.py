from datasets import load_dataset, load_from_disk, Dataset

dataset = load_from_disk("./uk_landcover_dataset")
dataset.push_to_hub("pints-sig/uk_landcover")