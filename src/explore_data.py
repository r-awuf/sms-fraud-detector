import pandas as pd

# Load data
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Drop the junk unnamed columns
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]) 

# Rename for clarity
df = df.rename(columns={"v1": "label", "v2": "message"})

print(df.head())
print(df.shape)

# Count spam vs ham
print("\n--- Class distribution ---")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True) * 100)

# Show sample messages
pd.set_option("display.max_colwidth", None)

print("\n--- Sample ham messages ---")
print(df[df["label"] == "ham"]["message"].sample(5, random_state=1).to_string(index=False))

print("\n--- Sample spam messages ---")
print(df[df["label"] == "spam"]["message"].sample(5, random_state=1).to_string(index=False))

# Add message length column
df["message_length"] = df["message"].apply(len)

print("\n--- Average message length by label ---")
print(df.groupby("label")["message_length"].mean())
