import pandas as pd

# Load your original CSV with UTF-8 encoding
df = pd.read_csv("HEAR_Dataset.csv", encoding="utf-8")

# Randomly sample 1000 rows
testing_df = df.sample(n=1000, random_state=42)

# Save the sampled dataset with UTF-8 encoding
testing_df.to_csv("testing_dataset.csv", index=False, encoding="utf-8-sig")

print("Random testing dataset of 1000 records saved as 'testing_dataset.csv' with proper Arabic text.")
