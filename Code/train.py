# train_t5.py
import argparse,os
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

def main(args):
    # Load dataset
    dataset = load_dataset("json", data_files=args.train_file)["train"]

    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    # Tokenize the input and output text
    def preprocess(example):
        input_enc = tokenizer(example["input"], padding="max_length", truncation=True, max_length=args.max_length)
        output_enc = tokenizer(example["output"], padding="max_length", truncation=True, max_length=args.max_length)
        input_enc["labels"] = output_enc["input_ids"]
        return input_enc
    print("TrainingArguments is from:", TrainingArguments.__module__)

    dataset = dataset.map(preprocess)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        report_to="none"
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model)
    )

    trainer.train()

    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument('--train_file', type=str, default=os.path.join(os.environ.get("SM_CHANNEL_TRAINING"), 'data.jsonl'))
    args = parser.parse_args()
    main(args)
