import json
import csv
import math
import torch
import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
import textstat

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Model paths
MODEL_PATHS = {
    "cd": "/Users/homitdalia/Documents/Repos/LLM-Privacy/models/cd/t5-cd-train-2025-04-12-21-50-16-999/output/model",
    "pbd": "/Users/homitdalia/Documents/Repos/LLM-Privacy/models/pbd/t5-pbd-train-2025-04-12-20-26-56-854/output/model"
}

# Load models and tokenizers
def load_model(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer_cd, model_cd = load_model(MODEL_PATHS["cd"])
tokenizer_pbd, model_pbd = load_model(MODEL_PATHS["pbd"])

# Metric functions
def redact(text):
    doc = nlp(text)
    redacted = text
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON", "EMAIL", "PHONE", "GPE", "LOC", "ORG"]:
            redacted = redacted[:ent.start_char] + "[REDACTED]" + redacted[ent.end_char:]
    return redacted

def generate_output(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            num_return_sequences=1,
            early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compute_bleu(reference, prediction):
    smoothing = SmoothingFunction().method4
    return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smoothing)

def compute_rouge_l(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

def compute_bertscore(reference, prediction):
    P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
    return F1[0].item()

def compute_repetition_penalty(prediction):
    tokens = prediction.lower().split()
    if not tokens:
        return 1.0
    unique_tokens = set(tokens)
    diversity = len(unique_tokens) / len(tokens)
    return 1.0 - diversity

def compute_meteor(reference, prediction):
    return meteor_score([reference.split()], prediction.split())

def compute_distinct_n(prediction, n=2):
    tokens = prediction.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngram_list = list(ngrams)
    if not ngram_list:
        return 0.0
    unique = len(set(ngram_list))
    return unique / len(ngram_list)

def compute_perplexity(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())

def evaluate_sample(prompt, reference):
    reference_redacted = redact(reference)
    output_cd = generate_output(prompt, tokenizer_cd, model_cd)
    output_pbd = generate_output(prompt, tokenizer_pbd, model_pbd)
    print("\n")
    print(f"CD Output: {output_cd}")
    print("\n")
    output_cd_redacted = redact(output_cd)
    print(f"Redacted CD Output: {output_cd_redacted}")
    print("\n")
    print(f"PBD Output: {output_pbd}")
    print("\n")
    return {
        "prompt": prompt,
        "reference": reference,
        "cd_output": output_cd,
        "pbd_output": output_pbd,

        "bleu_cd": compute_bleu(reference, output_cd),
        "bleu_pbd": compute_bleu(reference_redacted, output_pbd),

        "rouge_cd": compute_rouge_l(reference, output_cd),
        "rouge_pbd": compute_rouge_l(reference_redacted, output_pbd),

        "bertscore_cd": compute_bertscore(reference, output_cd),
        "bertscore_pbd": compute_bertscore(reference_redacted, output_pbd),

        "meteor_cd": compute_meteor(reference, output_cd),
        "meteor_pbd": compute_meteor(reference_redacted, output_pbd),

        "repetition_cd": compute_repetition_penalty(output_cd),
        "repetition_pbd": compute_repetition_penalty(output_pbd),

        "distinct1_cd": compute_distinct_n(output_cd, n=1),
        "distinct2_cd": compute_distinct_n(output_cd, n=2),
        "distinct1_pbd": compute_distinct_n(output_pbd, n=1),
        "distinct2_pbd": compute_distinct_n(output_pbd, n=2),

        "perplexity_cd": compute_perplexity(output_cd, tokenizer_cd, model_cd),
        "perplexity_pbd": compute_perplexity(output_pbd, tokenizer_pbd, model_pbd),

        "ref_perplexity_cd": compute_perplexity(reference, tokenizer_cd, model_cd),
        "ref_perplexity_pbd": compute_perplexity(reference_redacted, tokenizer_pbd, model_pbd)
    }

# Prompt user for mode
mode = input("Enter 'file' to evaluate all from unredacted.jsonl, or 'custom' for manual input: ").strip().lower()

results = []
if mode == "file":
    with open("/Users/homitdalia/Documents/Repos/LLM-Privacy/unredacted.jsonl", "r") as f, open("evaluation_results.csv", "w", newline='') as csvfile:
        writer = None
        for i, line in enumerate(f):
            print(f"Processing line {i + 1}")
            data = json.loads(line)
            prompt = data["input"]
            reference = data["output"]
            result = evaluate_sample(prompt, reference)
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                writer.writeheader()
            writer.writerow(result)
            results.append(result)

elif mode == "custom":
    prompt = input("Enter your prompt: ").strip()
    reference = """Jazmin Alvarez, a seasoned salesperson with an impressive track record of success, embarked on a significant job-related project in the past that showcased her exceptional skills and unwavering commitment to delivering remarkable results. The project involved securing a substantial sales contract with a prominent client, Acme Corporation, a leading provider of innovative technology solutions. Jazmin meticulously planned her approach, conducting extensive research on Acme Corporation's business objectives, challenges, and industry landscape. Armed with this knowledge, she tailored a compelling sales pitch that resonated with the client's unique requirements. Her persuasive communication style and in-depth understanding of the client's needs enabled her to establish a strong rapport and build trust. Throughout the sales cycle, Jazmin proactively addressed Acme Corporation's concerns and provided comprehensive solutions that aligned with their long-term goals. Her unwavering dedication and commitment to exceeding expectations resulted in a highly customized proposal that outlined a comprehensive strategy for achieving Acme Corporation's objectives. Jazmin's tenacity and unwavering focus on customer satisfaction played a pivotal role in closing the deal. She went the extra mile to provide exceptional support, ensuring that Acme Corporation had all the necessary information and resources to make an informed decision. Her exceptional negotiation skills further solidified the agreement, resulting in a mutually beneficial outcome for both parties. The successful completion of this project not only generated significant revenue for the company but also strengthened Jazmin's reputation as a highly skilled and dependable salesperson. Her ability to build strong relationships, provide personalized solutions, and consistently deliver exceptional results made her an invaluable asset to the sales team. Jazmin's unwavering commitment to excellence and her ability to consistently exceed client expectations have earned her a stellar reputation within the industry. She remains a highly sought-after salesperson, and her expertise and dedication continue to drive exceptional outcomes for her clients and the company. Contact Information: Phone Number: +86 13115 4183 Email Address: jazmin.alvarez4173@outlook.net"""
    result = evaluate_sample(prompt, reference)
    results.append(result)
    for k, v in result.items():
        print(f"{k}: {v}")

# import pandas as pd
# import ace_tools as tools; tools.display_dataframe_to_user(name="Evaluation Results", dataframe=pd.DataFrame(results))
