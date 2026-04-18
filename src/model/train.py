"""
CREATIVE Ad Generation Training Script
Fine-tunes Flan-T5-Small to generate CREATIVE marketing copy
With task prefixes and optimized for GPU training
"""
import os
import sys
import pandas as pd
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import mlflow
import mlflow.pytorch
from datetime import datetime
from accelerate import Accelerator
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

# Check GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎮 Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class Config:
    MODEL_NAME = "google/flan-t5-small"  # Instruction-tuned T5
    MAX_INPUT_LENGTH = 128
    MAX_TARGET_LENGTH = 128  # Perfect for ad copy
    BATCH_SIZE = 8 if DEVICE == "cuda" else 4  # Larger on GPU
    EPOCHS = 5  # Good for 1000 samples
    LEARNING_RATE = 3e-4
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = "ad-creative-generation"
    DATA_PATH = "data/products_sample.csv"

class HFWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = T5ForConditionalGeneration.from_pretrained(context.artifacts["model_path"])
        self.tokenizer = T5Tokenizer.from_pretrained(context.artifacts["model_path"])

    def predict(self, context, model_input):
        # model_input: list of strings
        inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = self.model.generate(**inputs, max_length=128)
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# Training uses real professional ad copy from CSV!


def create_ad_prompt(row):
    """Create prompt with task prefix for better T5 performance"""
    # CRITICAL: Task prefix helps Flan-T5 understand the task
    return f"""generate ad: Name: {row['product_name']}, Category: {row['category']}, Description: {row['description']}"""


def prepare_training_data(tokenizer, data_path):
    """Prepare dataset for ad generation training"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Create prompts in user's exact format
    df['input_text'] = df.apply(create_ad_prompt, axis=1)
    df['target_text'] = df['ad_copy']  # Use professional ad copy!
    
    print(f"\nTraining examples: {len(df)}")
    print("\nSample INPUT format:")
    print(df.iloc[0]['input_text'])
    print("\nSample OUTPUT (50-100 words):")
    print(df.iloc[0]['target_text'][:200] + "...")
    
    # Convert to Dataset
    dataset = Dataset.from_pandas(df[['input_text', 'target_text']])
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples['input_text'],
            max_length=Config.MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length"
        )
        
        labels = tokenizer(
            examples['target_text'],
            max_length=Config.MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 80/20 split
    split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"Train samples: {len(split['train'])}")
    print(f"Validation samples: {len(split['test'])}")
    
    return split['train'], split['test']


def train_model():
    """Train the creative ad generation model"""
    print("=" * 60)
    print("CREATIVE AD GENERATOR - Training")
    print("=" * 60)
    
    # MLflow setup
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"creative-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("model_name", Config.MODEL_NAME)
        mlflow.log_param("approach", "creative_fine_tuning")
        mlflow.log_param("batch_size", Config.BATCH_SIZE)
        mlflow.log_param("epochs", Config.EPOCHS)
        mlflow.log_param("learning_rate", Config.LEARNING_RATE)
        
        print(f"\nMLflow URI: {Config.MLFLOW_TRACKING_URI}")
        print(f"Experiment: {Config.MLFLOW_EXPERIMENT_NAME}")
        
        # Loading model
        print(f"\nLoading {Config.MODEL_NAME}...")
        tokenizer = T5Tokenizer.from_pretrained(Config.MODEL_NAME, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
        
        # Move model to GPU if available
        model = model.to(DEVICE)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model loaded: {params:.1f}M parameters on {DEVICE}")
        
        # Prepare data
        train_dataset, val_dataset = prepare_training_data(tokenizer, Config.DATA_PATH)
        
        # Training arguments (optimized for GPU if available)
        training_args = TrainingArguments(
            output_dir="./models/checkpoints",
            eval_strategy="epoch",  # Changed from evaluation_strategy
            save_strategy="epoch",
            learning_rate=Config.LEARNING_RATE,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.EPOCHS,
            weight_decay=0.01,
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=True if DEVICE == "cuda" else False,  # Mixed precision on GPU
            metric_for_best_model="loss",
            logging_dir="./logs",
            logging_steps=5,
            report_to=["none"],
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        print(f"\nStarting training...")
        print(f"  Epochs: {Config.EPOCHS}")
        print(f"  Batch size: {Config.BATCH_SIZE}")
        print(f"  Learning rate: {Config.LEARNING_RATE}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("\nThis will take 20-30 minutes on CPU. Please wait...")
        
        train_result = trainer.train()
        
        # Log metrics
        mlflow.log_metric("train_loss", train_result.training_loss)
        mlflow.log_metric("train_runtime_mins", train_result.metrics['train_runtime'] / 60)
        
        # Evaluate
        print("\nEvaluating...")
        eval_results = trainer.evaluate()
        # mlflow.log_metric("eval_loss", eval_results['eval_loss'])
        
        print(f"\nTraining completed!")
        print(f"  Final train loss: {train_result.training_loss:.4f}")
        print(f"  Final eval loss: {eval_results['eval_loss']:.4f}")
        print(f"  Training time: {train_result.metrics['train_runtime']/60:.1f} minutes")
        
        # Save model
        model_path = "./models/ad-creative-generator"
        print(f"\nSaving model to {model_path}...")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Testing with proper beam search (CRITICAL for quality!)
        print("\nTesting creative generation...")
        test_input = "generate ad: Name: Smart Coffee Maker, Category: Home & Kitchen, Description: WiFi-enabled coffee maker with app control"
        
        inputs = tokenizer(test_input, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # Move to GPU
        
        # PROPER generation parameters (as recommended)
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,           # Beam search for better quality
            early_stopping=True,   # Stop when done
            length_penalty=2.0,    # Encourage complete sentences
            no_repeat_ngram_size=2  # Prevent repetition
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nGenerated Ad:\n{generated}\n")
        
        # Save sample locally instead of MLflow (avoids credentials error)
        with open("sample_generation.txt", "w", encoding="utf-8") as f:
            f.write(generated)
        
        # Log model to MLflow (try-except to handle MinIO issues)
        print("Logging to MLflow...")
        try:
            mlflow.log_artifacts(model_path, artifact_path="ad-creative-generator")
            # run_id = mlflow.active_run().info.run_id
            # mlflow.register_model(f"runs:/{run_id}/ad-creative-generator", "ad-creative-generator")

            # acc = Accelerator()
            # model_acc = acc.unwrap_model(model)
            mlflow.pyfunc.log_model(
                artifact_path="ad-creative-generator",
                python_model=HFWrapper(),
                artifacts={"model_path": model_path}
            )
            print("Model logged to MLflow registry successfully!")
        except Exception as e:
            print(f"Warning: Could not log model to MLflow: {e}")
            print("Model is still saved locally in ./models/ad-creative-generator")

        
        print("\nSUCCESS! Model ready for deployment.")
        print(f"Check MLflow UI: {Config.MLFLOW_TRACKING_URI}")
        
        return model, tokenizer


if __name__ == "__main__":
    try:
        model, tokenizer = train_model()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    # print("Starting training script...")