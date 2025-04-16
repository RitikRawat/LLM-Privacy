import pandas as pd
import streamlit as st

# Set layout
st.set_page_config(layout="wide")
st.title("📊 LLM Privacy Evaluation Viewer")

# Load datasets
df = pd.read_csv("evaluation_results.csv")
df_pii = pd.read_csv("cd_pii_leakage.csv")

# Merge PII leakage data if not already merged
if "cd_pii_count" not in df.columns and "cd_pii_count" in df_pii.columns:
    df["cd_pii_count"] = df_pii["cd_pii_count"]

# Sidebar list view
st.sidebar.header("🔍 Select a Sample")
selected_idx = st.sidebar.radio(
    "Samples",
    options=df.index,
    format_func=lambda i: f"{i}: {df['prompt'][i][:80]}..."
)

# Extract selected row
row = df.loc[selected_idx]

# Display prompt and outputs
st.markdown("### 📝 Prompt")
st.info(row["prompt"])

st.markdown("### 🎯 Reference")
st.success(row["reference"])

st.markdown("### 🤖 CD Output")
st.warning(row["cd_output"])
st.markdown("The metrics were calculated after redacting the reference output. The PII leakage count and the output above is based on the original output.")


st.markdown("### 🛡️ PBD Output")
st.warning(row["pbd_output"])

# Evaluation metrics including PII
st.markdown("### 📊 Evaluation Scores")

metrics_data = {
    "Metric": [
        "BLEU", "ROUGE-L", "BERTScore", "METEOR", "Repetition",
        "Distinct-1", "Distinct-2", "Perplexity", "Ref Perplexity", "PII Leakage (CD)"
    ],
    "CD": [
        row["bleu_cd"], row["rouge_cd"], row["bertscore_cd"], row["meteor_cd"], row["repetition_cd"],
        row["distinct1_cd"], row["distinct2_cd"], row["perplexity_cd"], row["ref_perplexity_cd"], row.get("cd_pii_count", "N/A")
    ],
    "PBD": [
        row["bleu_pbd"], row["rouge_pbd"], row["bertscore_pbd"], row["meteor_pbd"], row["repetition_pbd"],
        row["distinct1_pbd"], row["distinct2_pbd"], row["perplexity_pbd"], row["ref_perplexity_pbd"], "-"
    ]
}

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")
