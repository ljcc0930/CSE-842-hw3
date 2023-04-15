from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

train_data = dataset["train"].shuffle(seed=1).select(range(1000))
test_data = dataset["test"].shuffle(seed=1).select(range(1000))

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
test_data = test_data.map(tokenize, batched=True, batch_size=len(test_data))
train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

trainer.train()
trainer.evaluate()