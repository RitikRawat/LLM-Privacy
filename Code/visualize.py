import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load the evaluation results
file_path = "evaluation_results.csv"
if not os.path.exists(file_path):
    print(f"❌ File '{file_path}' not found.")
    exit(1)

df = pd.read_csv(file_path)

# Create output directory
os.makedirs("charts", exist_ok=True)

# Set global style
sns.set(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6)})

# === Metric Comparisons (Box Plots) ===
metrics = [
    ("bleu_cd", "bleu_pbd"),
    ("rouge_cd", "rouge_pbd"),
    ("bertscore_cd", "bertscore_pbd"),
    ("meteor_cd", "meteor_pbd"),
    ("repetition_cd", "repetition_pbd"),
    ("distinct1_cd", "distinct1_pbd"),
    ("distinct2_cd", "distinct2_pbd"),
    ("perplexity_cd", "perplexity_pbd"),
    ("ref_perplexity_cd", "ref_perplexity_pbd")
]

for metric_cd, metric_pbd in metrics:
    metric_name = metric_cd.replace("_cd", "").upper()
    data = pd.DataFrame({
        "CD": df[metric_cd],
        "PBD": df[metric_pbd]
    })

    plt.figure()
    sns.boxplot(data=data, palette="Set2")
    sns.stripplot(data=data, color='black', size=2, alpha=0.3, jitter=True)
    plt.title(f"{metric_name} Distribution (CD vs PBD)")
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(f"charts/{metric_name.lower()}_boxplot.png", dpi=500)
    plt.close()

# === Summary Bar Chart (Mean ± Std) ===
metric_names = []
cd_means = []
cd_stds = []
pbd_means = []
pbd_stds = []

for metric_cd, metric_pbd in metrics:
    name = metric_cd.replace("_cd", "").upper()
    metric_names.append(name)
    cd_means.append(df[metric_cd].mean())
    cd_stds.append(df[metric_cd].std())
    pbd_means.append(df[metric_pbd].mean())
    pbd_stds.append(df[metric_pbd].std())

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, cd_means, width, yerr=cd_stds, capsize=5, label='CD', color='salmon')
plt.bar(x + width/2, pbd_means, width, yerr=pbd_stds, capsize=5, label='PBD', color='skyblue')
plt.xticks(x, metric_names, rotation=45)
plt.ylabel("Mean Metric Value")
plt.title("CD vs PBD: Mean ± Std of Each Metric")
plt.legend()
plt.tight_layout()
plt.savefig("charts/summary_bar_chart.png", dpi=500)
plt.close()

# === Privacy vs Utility Chart ===
plt.figure(figsize=(10, 6))
plt.scatter(df["ref_perplexity_cd"], df["bertscore_cd"], label="CD", color="red", alpha=0.6)
plt.scatter(df["ref_perplexity_pbd"], df["bertscore_pbd"], label="PBD", color="blue", alpha=0.6)

plt.title("Privacy vs Utility Tradeoff")
plt.xlabel("Privacy (↑ Ref Perplexity)")
plt.ylabel("Utility (↑ BERTScore)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/privacy_vs_utility.png", dpi=500)
plt.close()

print("✅ All updated charts saved to the 'charts' directory.")
