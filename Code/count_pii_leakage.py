import pandas as pd
import spacy
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the original CSV
input_path = "evaluation_results.csv"
if not os.path.exists(input_path):
    print(f"❌ File '{input_path}' not found.")
    exit(1)

df = pd.read_csv(input_path)

# Detect and count PII in cd_output
def count_pii(text):
    doc = nlp(str(text))
    pii_labels = ["PERSON", "EMAIL", "PHONE", "GPE", "LOC", "ORG"]
    return sum(1 for ent in doc.ents if ent.label_ in pii_labels)

# Apply to the cd_output column
df["cd_pii_count"] = df["cd_output"].apply(count_pii)

# Save just the count column to new file
df[["cd_pii_count"]].to_csv("cd_pii_leakage.csv", index=False)

# Calculate the average PII leakage count
average_pii_leakage = df["cd_pii_count"].mean()
print(f"📊 Average PII leakage count: {average_pii_leakage}")

print("✅ PII leakage count saved to 'cd_pii_leakage.csv'")
