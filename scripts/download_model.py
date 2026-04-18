"""
Quick test script to verify we can use the model without full training
Downloads Flan-T5-Small and saves it locally for API testing
"""
import os
import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration

print("Downloading Flan-T5-Small model...")

model_name = "google/flan-t5-small"
save_path = "./models/ad-creative-generator"

# Create directory
os.makedirs(save_path,exist_ok=True)

# Download and save
print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.save_pretrained(save_path)

print(f"Model saved to {save_path}")
print("Testing generation...")

# Quick test
test_input = "Generate a compelling advertisement for: Wireless Bluetooth Headphones. Category: Electronics. Description: Premium noise-cancelling headphones. Price: $299.99."
inputs = tokenizer(test_input, return_tensors="pt", max_length=128, truncation=True)
outputs = model.generate(**inputs, max_length=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nTest Input: {test_input[:80]}...")
print(f"\nTest Output: {result}")
print("\nSUCCESS! Model ready for API testing.")
