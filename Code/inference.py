# inference.py

import json
import torch
import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load spaCy English model once globally
nlp = spacy.load("en_core_web_sm")

# Load model and tokenizer from SageMaker model_dir
def model_fn(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

# Redact detected PII entities
def redact_text(text):
    doc = nlp(text)
    redacted = text
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON", "EMAIL", "PHONE", "GPE", "LOC", "ORG"]:
            redacted = redacted[:ent.start_char] + "[REDACTED]" + redacted[ent.end_char:]
    return redacted

# Generate T5 output
def predict_fn(input_data, model_and_tokenizer):
    tokenizer, model = model_and_tokenizer

    # Get the input text and admin flag
    prompt = input_data.get("inputs", "")
    is_admin = input_data.get("admin", False)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model.generate(
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

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if is_admin:
        return {"raw": raw_output}
    else:
        redacted_output = redact_text(raw_output)
        return {"redacted": redacted_output}
