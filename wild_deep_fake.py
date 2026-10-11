from datasets import load_dataset


# 1. Load the dataset from the Hugging Face Hub
dataset = load_dataset("xingjunm/WildDeepfake")


# 2. Save it to your local directory
dataset.save_to_disk("path/to/local/directory")

