
from pathlib import Path
import kagglehub

# Download the dataset from Kaggle
path = Path(kagglehub.dataset_download("yasserh/housing-prices-dataset"))

print("Dataset downloaded to:", path)

# List the files we downloaded
print("\nFiles found:")

for file in path.rglob("*"):
    if file.is_file():
        print(file.name, "->", file)
