from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

# Load Pre-Trained CodeT5 Model and Tokenizer
model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load and Prepare a Smaller Dataset Subset
dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
train_subset = dataset["train"].shuffle(seed=42).select(range(10000))  # 10k samples
val_subset = dataset["validation"].shuffle(seed=42).select(range(1000))  # 1k samples

# Preprocess the Dataset
def preprocess_function(examples):
    inputs = [f"Generate Python code: {desc}" for desc in examples["func_documentation_string"]]
    targets = examples["func_code_string"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_train = train_subset.map(preprocess_function, batched=True)
tokenized_val = val_subset.map(preprocess_function, batched=True)

# Define Training Arguments for Lightweight Fine-Tuning
training_args = Seq2SeqTrainingArguments(
    output_dir="./codet5-python-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,  
    num_train_epochs=1,  
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # Mixed-precision training
    predict_with_generate=True,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

trainer.train()

# Save the Fine-Tuned Model
model.save_pretrained("./codet5-python-finetuned")
tokenizer.save_pretrained("./codet5-python-finetuned")

print("Lightweight fine-tuning complete. Saved to './codet5-python-finetuned'")
