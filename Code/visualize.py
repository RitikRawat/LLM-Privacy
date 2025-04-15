import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the evaluation results
file_path = "evaluation_results.csv"
if not os.path.exists(file_path):
    print(f"❌ File '{file_path}' not found.")
    exit(1)

df = pd.read_csv(file_path)

# Create output directory
os.makedirs("charts", exist_ok=True)

# === Metric Comparisons: CD vs PBD ===
plt.style.use("ggplot")
plt.rcParams.update({'figure.figsize': (12, 6), 'axes.grid': True})

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
    plt.figure()
    plt.plot(df[metric_cd], label="CD", marker='o')
    plt.plot(df[metric_pbd], label="PBD", marker='x')
    plt.title(f"{metric_name} Comparison (CD vs PBD)")
    plt.xlabel("Sample Index")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"charts/{metric_name.lower()}_comparison.png", dpi=500)
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

print("✅ All charts saved to the 'charts' directory.")
