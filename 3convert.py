
#Step 3: Convert to GGUF Format
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import LlamaConverter

def convert_to_gguf(model_path, output_path, tokenizer_path):
    print(f"[INFO] Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

    print(f"[INFO] Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"[INFO] Converting to GGUF format...")
    converter = LlamaConverter(model=model, tokenizer=tokenizer)
    converter.convert(output_path)

    print(f"[SUCCESS] GGUF model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--output-path", required=True, help="Path to save GGUF file")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer directory or file")
    args = parser.parse_args()

    convert_to_gguf(args.model_path, args.output_path, args.tokenizer_path)