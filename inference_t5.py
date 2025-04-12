# inference_t5.py
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.3,
        num_return_sequences=1,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    print('here')
    while True:
        user_input = input("Ask a question:\n")
        response = generate_response(user_input, tokenizer, model)
        print(f"\nResponse:\n{response}\n")
