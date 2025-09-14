# Step 2: Merge LoRA Adapter into Base Mode
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = Path("tinylama-php-crm/checkpoint-195").resolve()
output_path = Path("tinylama-php-crm-merged").resolve()

# Load base + adapter
base = AutoModelForCausalLM.from_pretrained(base_model_id)
peft = PeftModel.from_pretrained(base, str(adapter_path), local_files_only=True)

# Merge and save
merged = peft.merge_and_unload()
merged.save_pretrained(str(output_path))

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(str(output_path))

print(f"[SUCCESS] Merged model and tokenizer saved to: {output_path}")