import torch
import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import re
from nltk.translate.meteor_score import meteor_score
import textstat
import torch.nn.functional as F
import math

# import nltk
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Optional but improves METEOR

# Load spaCy model for redaction
nlp = spacy.load("en_core_web_sm")

# Model paths
MODEL_PATHS = {
    "cd": "/Users/homitdalia/Documents/Repos/LLM-Privacy/models/cd/t5-cd-train-2025-04-12-21-50-16-999/output/model",
    "pbd": "/Users/homitdalia/Documents/Repos/LLM-Privacy/models/pbd/t5-pbd-train-2025-04-12-20-26-56-854/output/model"
}

# Load model and tokenizer
def load_model(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# Redact PII
def redact(text):
    doc = nlp(text)
    redacted = text
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON", "EMAIL", "PHONE", "GPE", "LOC", "ORG"]:
            redacted = redacted[:ent.start_char] + "[REDACTED]" + redacted[ent.end_char:]
    return redacted

# Run inference
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

# Calculate BLEU
def compute_bleu(reference, prediction):
    smoothing = SmoothingFunction().method4
    return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smoothing)

def compute_rouge_l(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure  # Return F1 score

def compute_bertscore(reference, prediction):
    P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
    return F1[0].item()  # Return first (and only) example

def compute_repetition_penalty(prediction):
    tokens = prediction.lower().split()
    if not tokens:
        return 1.0  # Max penalty
    unique_tokens = set(tokens)
    diversity = len(unique_tokens) / len(tokens)
    return 1.0 - diversity  # Higher means more repetitive

def compute_meteor(reference, prediction):
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    return meteor_score([reference_tokens], prediction_tokens)

def compute_distinct_n(prediction, n=2):
    tokens = prediction.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngram_list = list(ngrams)
    if not ngram_list:
        return 0.0
    unique = len(set(ngram_list))
    return unique / len(ngram_list)

def compute_readability(text):
    return textstat.flesch_reading_ease(text)  # Higher is easier

def compute_perplexity(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())

# === MAIN ===
if __name__ == "__main__":
    # Example prompt — ideally tied to known references
    prompt = input("Enter your prompt: ").strip()

    # Load both models
    tokenizer_cd, model_cd = load_model(MODEL_PATHS["cd"])
    tokenizer_pbd, model_pbd = load_model(MODEL_PATHS["pbd"])

    # Generate predictions
    output_cd = generate_output(prompt, tokenizer_cd, model_cd)
    output_pbd = generate_output(prompt, tokenizer_pbd, model_pbd)

    print("\n--- Model Outputs ---")
    print("[CD Model]:")
    print(output_cd)
    print("\n[PBD Model]:")
    print(output_pbd)

    # Example references (tie these to prompt in real use)
    reference_unredacted = """My name is Aaliyah Popova, and I am a jeweler with 13 years of experience. I remember a very unique and challenging project I had to work on last year. A customer approached me with a precious family heirloom - a diamond necklace that had been passed down through generations. Unfortunately, the necklace was in poor condition, with several loose diamonds and a broken clasp. The customer wanted me to restore it to its former glory, but it was clear that this would be no ordinary repair. Using my specialized tools and techniques, I began the delicate task of dismantling the necklace. Each diamond was carefully removed from its setting, and the damaged clasp was removed. Once the necklace was completely disassembled, I meticulously cleaned each diamond and inspected it for any damage. Fortunately, the diamonds were all in good condition, with no cracks or chips. The next step was to repair the broken clasp. I carefully soldered the broken pieces back together, ensuring that the clasp was sturdy and secure. Once the clasp was repaired, I began the process of reassembling the necklace. Each diamond was carefully placed back into its setting, and the necklace was polished until it sparkled like new. When I presented the restored necklace to the customer, they were overjoyed. They couldn't believe that I had been able to bring their family heirloom back to life. The necklace looked as beautiful as it had when it was first created, and the customer was thrilled to have it back in their possession. If you have a project that you would like to discuss, please feel free to contact me by phone at (95) 94215-7906 or by email at aaliyah.popova4783@aol.edu. I look forward to hearing from you! P.S.: When I'm not creating beautiful jewelry, I enjoy spending time podcasting. I love sharing my knowledge about jewelry and connecting with other people who are passionate about this art form. I also enjoy spending time with my family and exploring new places. If you would like to learn more about me, please feel free to visit my website at [website address] or visit me at my studio located at 97 Lincoln Street.""".strip()
    reference_redacted = redact(reference_unredacted)

    # Compute BLEU
    bleu_cd = compute_bleu(reference_unredacted, output_cd)
    bleu_pbd = compute_bleu(reference_redacted, output_pbd)

    print("\n--- Evaluation Metrics ---")

    print("\n--- BLEU Scores ---")
    print(f"BLEU (CD):  {bleu_cd:.4f}")
    print(f"BLEU (PBD): {bleu_pbd:.4f}")

    # CD Model Metrics
    print("[CD Model]")
    print(f"BLEU:       {compute_bleu(reference_unredacted, output_cd):.4f}")
    print(f"ROUGE-L:    {compute_rouge_l(reference_unredacted, output_cd):.4f}")
    print(f"BERTScore:  {compute_bertscore(reference_unredacted, output_cd):.4f}")
    print(f"Repetition: {compute_repetition_penalty(output_cd):.4f}")

    # PBD Model Metrics
    print("\n[PBD Model]")
    print(f"BLEU:       {compute_bleu(reference_redacted, output_pbd):.4f}")
    print(f"ROUGE-L:    {compute_rouge_l(reference_redacted, output_pbd):.4f}")
    print(f"BERTScore:  {compute_bertscore(reference_redacted, output_pbd):.4f}")
    print(f"Repetition: {compute_repetition_penalty(output_pbd):.4f}")

    # Compute METEOR
    print("\n--- METEOR Scores ---")
    print(f"METEOR (CD):  {compute_meteor(reference_unredacted, output_cd):.4f}")
    print(f"METEOR (PBD): {compute_meteor(reference_redacted, output_pbd):.4f}")
    print(f"METEOR (Reference): {compute_meteor(reference_redacted, reference_unredacted):.4f}")
    print(f"METEOR (Redacted): {compute_meteor(reference_redacted, reference_redacted):.4f}")
    
    distinct1 = compute_distinct_n(output_cd, n=1)
    distinct2 = compute_distinct_n(output_cd, n=2)
    print(f"Distinct-1 (CD): {distinct1:.4f}")
    print(f"Distinct-2 (CD): {distinct2:.4f}")
    
    distinct1 = compute_distinct_n(output_pbd, n=1)
    distinct2 = compute_distinct_n(output_pbd, n=2)
    print(f"Distinct-1 (PBD): {distinct1:.4f}")
    print(f"Distinct-2 (PBD): {distinct2:.4f}")

    # Perplexity evaluation
    ppl_cd = compute_perplexity(output_cd, tokenizer_cd, model_cd)
    ppl_pbd = compute_perplexity(output_pbd, tokenizer_pbd, model_pbd)

    print("\n--- Perplexity ---")
    print(f"Perplexity (CD):  {ppl_cd:.4f}")
    print(f"Perplexity (PBD): {ppl_pbd:.4f}")

    ref_ppl_cd = compute_perplexity(reference_unredacted, tokenizer_cd, model_cd)
    ref_ppl_pbd = compute_perplexity(reference_redacted, tokenizer_pbd, model_pbd)

    print("\n--- Perplexity of Reference ---")
    print(f"Ref Perplexity (CD model on unredacted):  {ref_ppl_cd:.4f}")
    print(f"Ref Perplexity (PBD model on redacted):   {ref_ppl_pbd:.4f}")

