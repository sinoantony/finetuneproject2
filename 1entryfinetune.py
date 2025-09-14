#Step 1: Fine-Tune with LoRA and 4-bit Quantization on PHP CRM Codebase

import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType

# === Step 1: Collect PHP Files ===
def collect_php_files(root_dir):
    return [
        os.path.join(subdir, file)
        for subdir, _, files in os.walk(root_dir)
        for file in files if file.endswith(".php")
    ]

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# === Step 2: Format for Instruction Tuning ===
def format_instruction(code):
    return {
        "prompt": f"<|user|>\nExplain this PHP code:\n{code}\n<|assistant|>\n",
        "response": "This PHP code defines CRM functionality. It likely handles customer data, transactions, and business logic."
    }

def format_dataset(file_paths):
    return [format_instruction(read_file(path)) for path in file_paths]

# === Step 3: Tokenize Dataset ===
def tokenize_dataset(dataset, tokenizer):
    def tokenize(example):
        tokens = tokenizer(
            example["prompt"] + example["response"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    return dataset.map(tokenize)

# === Step 4: Load Model with LoRA + Quantization ===
def load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(base_model, peft_config)
    return tokenizer, model

# === Step 5: Train the Model ===
def train_model(model, tokenizer, dataset):
    tokenized_ds = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir="./tinylama-php-crm",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        eval_strategy="epoch",
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"]
    )

    trainer.train()

# === Step 6: Main Execution ===
def main():
    print("[INFO] Collecting PHP files...")
    php_files = collect_php_files("./crm-php/")
    print(f"[INFO] Found {len(php_files)} PHP files.")

    if not php_files:
        print("[⚠️] No PHP files found. Please check the directory path.")
        return

    print("[INFO] Formatting dataset...")
    formatted_data = format_dataset(php_files)
    hf_dataset = Dataset.from_list(formatted_data).train_test_split(test_size=0.1)

    print("[INFO] Loading TinyLlama with LoRA...")
    tokenizer, model = load_model()

    print("[INFO] Starting training...")
    train_model(model, tokenizer, hf_dataset)

    print("[✅] Fine-tuning complete. Model saved to ./tinylama-php-crm")

if __name__ == "__main__":
    main()