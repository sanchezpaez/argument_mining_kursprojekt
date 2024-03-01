import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer

from evaluate import generate_classification_report
from preprocess_corpus import load_data


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


class InputFeatures:
    def __init__(self, input_ids, attention_mask, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label


def prepare_datasets(X_train, y_train, X_val, y_val, tokenizer, unique_labels):
    label_map = {label: i for i, label in enumerate(unique_labels)}
    labels_numeric_train = [label_map[label] for label in y_train]
    labels_numeric_val = [label_map[label] for label in y_val]

    encodings_train = tokenizer(X_train, truncation=True, padding=True)
    encodings_val = tokenizer(X_val, truncation=True, padding=True)

    train_features = []
    for i in range(len(X_train)):
        train_features.append(InputFeatures(input_ids=encodings_train['input_ids'][i],
                                      attention_mask=encodings_train['attention_mask'][i],
                                      label=labels_numeric_train[i]))

    val_features = []
    for i in range(len(X_val)):
        val_features.append(InputFeatures(input_ids=encodings_val['input_ids'][i],
                                            attention_mask=encodings_val['attention_mask'][i],
                                            label=labels_numeric_val[i]))

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

    return train_dataset, val_dataset, tokenizer, label_map


def train_model(model, train_dataset, val_dataset, training_args):
    data_collator = CustomDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()

    return trainer


def evaluate_model(trainer, val_dataset, all_labels, label_map):
    print("Type of trainer:", type(trainer))
    print("Type of val_dataset:", type(val_dataset))
    # Generate predictions using the trained model
    predictions = trainer.predict(val_dataset)
    print("Type of predictions:", type(predictions))

    # Extract predicted labels from the predictions
    predicted_labels_numeric = predictions.predictions.argmax(axis=1)
    print("Type of predicted_labels_numeric:", type(predicted_labels_numeric))

    # Convert predicted numeric labels to their corresponding string labels
    predicted_labels = [all_labels[label] for label in predicted_labels_numeric]
    print("Type of predicted_labels:", type(predicted_labels))

    # Extract actual labels from the validation dataset
    actual_labels = [all_labels[label] for label in val_dataset.tensors[2].tolist()]

    # Convert actual labels to their corresponding numeric labels using label map
    actual_labels_numeric = [label_map[label] for label in actual_labels]

    # Calculate accuracy using actual and predicted labels
    accuracy = accuracy_score(actual_labels_numeric, predicted_labels_numeric)

    # Return accuracy and other evaluation metrics
    return accuracy, actual_labels, predicted_labels, actual_labels_numeric





if __name__ == "__main__":
    X_train = load_data('X_train.pkl')
    y_train = load_data('y_train.pkl')
    X_dev = load_data('X_dev.pkl')
    y_dev = load_data('y_dev.pkl')
    X_test = load_data('X_test.pkl')
    y_test = load_data('y_test.pkl')
    mlb = load_data('mlb.pkl')
    all_labels = load_data('unique_labels.pkl')

    X_train = X_train[:10]
    # print(X_train)
    y_train = y_train[:10]
    X_dev = X_dev[:10]
    y_dev = y_dev[:10]
    # print(texts)
    # print(labels)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset, val_dataset, tokenizer, label_map = prepare_datasets(X_train, y_train, X_dev, y_dev, tokenizer, all_labels)
    print(all_labels)

    # all_labels = list(set(labels))

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(all_labels))

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = train_model(model, train_dataset, val_dataset, training_args)

    accuracy, actual_labels, predicted_labels, actual_labels_numeric = evaluate_model(trainer, val_dataset, all_labels,
                                                                                      label_map)
    unique_labels = set(actual_labels)
    num_classes = len(unique_labels)

    print("Accuracy:", accuracy)  # For the first 100 Accuracy: 0.65 no features
    # 1000 acc 0.695



    if len(unique_labels) != num_classes:
        print("Number of unique labels:", len(unique_labels))
        print("Expected number of classes:", num_classes)
        print("Actual labels:", unique_labels)
        print("Predicted labels:", predicted_labels)
        print("All labels:", all_labels)

    report = generate_classification_report(actual_labels_numeric, [label_map[label] for label in predicted_labels],
                                            all_labels)
