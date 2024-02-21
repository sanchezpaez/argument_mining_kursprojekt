import numpy as np
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
import torch
from sklearn.metrics import accuracy_score, classification_report

# Define a custom data collator
class CustomDataCollator(torch.utils.data.DataLoader):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        if isinstance(features[0], tuple):
            batch['input_ids'] = torch.stack([f[0] for f in features])
            batch['attention_mask'] = torch.stack([f[1] for f in features])
            batch['labels'] = torch.tensor([f[2] for f in features], dtype=torch.long)
        else:
            batch['input_ids'] = torch.stack([torch.tensor(f.input_ids, dtype=torch.long) for f in features])
            batch['attention_mask'] = torch.stack([torch.tensor(f.attention_mask, dtype=torch.long) for f in features])
            batch['labels'] = torch.tensor([f.label for f in features], dtype=torch.long)
        return batch

# Define InputFeatures class
class InputFeatures:
    def __init__(self, input_ids, attention_mask, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label


# Different texts with corresponding unique labels
texts = [
    "This is a positive review.",
    "This is a negative review.",
    "This is a neutral review.",
    "This is a mixed review."
]

labels = [
    "positive",
    "negative",
    "neutral",
    "mixed"
]

all_labels = ["positive", "negative", "neutral", "mixed"]

# Mapping labels to numeric values
label_map = {"positive": 0, "negative": 1, "neutral": 2, "mixed": 3}
labels_numeric = [label_map[label] for label in labels]

# Load tokenizer and tokenize data
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
encodings = tokenizer(texts, truncation=True, padding=True)

# Convert encodings to InputFeatures
features = []
for i in range(len(texts)):
    features.append(InputFeatures(input_ids=encodings['input_ids'][i],
                                  attention_mask=encodings['attention_mask'][i],
                                  label=labels_numeric[i]))

# Split dataset into train and validation sets
train_features, val_features = train_test_split(features, test_size=0.2, random_state=42)

# Prepare PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor([f.input_ids for f in train_features]),
    torch.tensor([f.attention_mask for f in train_features]),
    torch.tensor([f.label for f in train_features])
)

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor([f.input_ids for f in val_features]),
    torch.tensor([f.attention_mask for f in val_features]),
    torch.tensor([f.label for f in val_features])
)

# Load the model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_map))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Custom data collator
data_collator = CustomDataCollator(tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=val_dataset)

# Get predictions
predictions = trainer.predict(val_dataset)

# Extract predicted labels and convert to numpy array
predicted_labels = predictions.predictions.argmax(axis=1)

# Convert predicted labels to original label names
predicted_labels = [all_labels[label] for label in predicted_labels]
predicted_labels_numeric = [label_map[label] for label in predicted_labels]
print(f"The predicted labels are {predicted_labels}")

# Get actual labels
actual_labels = [all_labels[label] for label in val_dataset.tensors[2].tolist()]
print(f"The actual labels are {actual_labels}")

# Convert actual labels to their corresponding numeric values
actual_labels_numeric = [label_map[label] for label in actual_labels]

# Calculate accuracy
accuracy = accuracy_score(actual_labels_numeric, predicted_labels)
print("Accuracy:", accuracy)

# Count the unique values in the actual labels
unique_labels = set(actual_labels)
num_classes = len(all_labels)

# Check if all classes are present
if len(unique_labels) != num_classes:
    print("Number of unique labels:", len(unique_labels))
    print("Expected number of classes:", num_classes)
    print("Actual labels:", unique_labels)
    print("All labels:", all_labels)

# Generate classification report
class_names = ["positive", "negative", "neutral", "mixed"]  # Update with your class names
report = classification_report(actual_labels_numeric, predicted_labels_numeric, labels=np.arange(0, len(all_labels)), target_names=class_names, digits=4, zero_division=0)
print("Classification Report:\n", report)

