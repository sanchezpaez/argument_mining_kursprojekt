import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from evaluate import generate_classification_report
from preprocess_corpus import load_data


class CustomDataCollator:
    def __call__(self, features):
        print("Number of features:", len(features))
        print("Feature example:", features[0])
        # Prepare input tensors
        input_ids = torch.stack([f[0] for f in features], dim=0)
        print("Input IDs shape:", input_ids.shape)
        attention_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
        print("Attention mask shape:", attention_mask.shape)

        # Prepare labels tensor
        # Extract labels from the third element of each feature and convert to tensor
        labels = [torch.tensor(f[2], dtype=torch.long) for f in features]
        print("Labels length:", len(labels))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class InputFeatures:
    def __init__(self, input_ids, attention_mask, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label


def prepare_datasets(X_train, y_train, X_val, y_val, tokenizer, unique_labels):
    # Flatten y_train and y_val
    y_train_flat = [label for sublist in y_train for label in sublist]
    y_val_flat = [label for sublist in y_val for label in sublist]

    # Create label mapping
    label_map = {label: i for i, label in enumerate(unique_labels)}

    # Convert multi-label format to single-label format for training data
    y_train_single = [label_map[label] for label in y_train_flat]

    # Convert multi-label format to single-label format for validation data
    y_val_single = [label_map[label] for label in y_val_flat]

    # Tokenize training and validation data
    encodings_train = tokenizer(X_train, truncation=True, padding=True)
    encodings_val = tokenizer(X_val, truncation=True, padding=True)

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings_train['input_ids']),
        torch.tensor(encodings_train['attention_mask']),
        torch.tensor(y_train_single, dtype=torch.long)  # Ensure labels are of type long
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings_val['input_ids']),
        torch.tensor(encodings_val['attention_mask']),
        torch.tensor(y_val_single, dtype=torch.long)  # Ensure labels are of type long
    )

    return train_dataset, val_dataset, tokenizer, label_map


def prepare_dataset(X, y, tokenizer, mlb=None, max_label_length=17):
    encodings = tokenizer(X, truncation=True, padding=True)
    if mlb:
        labels = mlb.inverse_transform(y)
        print(labels)
    else:
        labels = y
        print(labels)

    encoded_labels = []
    for label_list in labels:
        encoded_label = tokenizer.encode(", ".join(label_list), add_special_tokens=False)
        # Pad or truncate the encoded label to ensure consistent length
        encoded_label = encoded_label[:max_label_length] + [tokenizer.pad_token_id] * (max_label_length - len(encoded_label))
        encoded_labels.append(encoded_label)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings['input_ids']),
        torch.tensor(encodings['attention_mask']),
        torch.tensor(encoded_labels, dtype=torch.long)  # Ensure labels are of type long
    )

    return dataset



def train_model(model, train_dataset, val_dataset, training_args):
    data_collator = CustomDataCollator()

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
    print(type(mlb))
    all_labels = load_data('unique_labels.pkl')
    print('The unique labels are:', all_labels)

    X_train = X_train[:10]
    print('X_train:', X_train)
    y_train = y_train[:10]
    print('y_train:', y_train)
    X_dev = X_dev[:10]
    y_dev = y_dev[:10]
    # print(texts)
    # print(labels)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # train_dataset, val_dataset, tokenizer, label_map = prepare_datasets(X_train, y_train, X_dev, y_dev, tokenizer, all_labels)
    train_dataset = prepare_dataset(X_train, y_train, tokenizer, mlb)
    val_dataset = prepare_dataset(X_dev, y_dev, tokenizer, mlb)

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
