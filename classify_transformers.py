import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from evaluate import generate_classification_report
from classify_rfc import load_data


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


def recover_data():
    X_train = load_data('X_train.pkl')
    print('Items in X_train:', len(X_train))
    y_train = load_data('y_train.pkl')
    print('Items in y_train:', len(y_train))
    X_dev = load_data('X_dev.pkl')
    print('Items in X_dev:', len(X_dev))
    y_dev = load_data('y_dev.pkl')
    print('Items in y_dev:', len(y_dev))
    X_test = load_data('X_test.pkl')
    print('Items in X_test:', len(X_test))
    y_test = load_data('y_test.pkl')
    print('Items in y_test:', len(y_test))
    unique_labels = load_data('unique_labels.pkl')
    print(f'There are {len(unique_labels)} unique labels')
    print('The unique labels are:', unique_labels)

    return X_train, y_train, X_dev, y_dev, X_test, y_test, unique_labels




def prepare_datasets(X_train, y_train, X_val, y_val, labels):
    print('Preparing datasets as tensors...')
    label_map = {label: i for i, label in enumerate(labels)}

    labels_numeric_train = [label_map[label] for label in y_train]
    labels_numeric_val = [label_map[label] for label in y_val]

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    val_encodings = tokenizer(X_val, truncation=True, padding=True)

    train_features = []
    for i in range(len(X_train)):
        train_features.append(InputFeatures(input_ids=train_encodings['input_ids'][i],
                                            attention_mask=train_encodings['attention_mask'][i],
                                            label=labels_numeric_train[i]))

    val_features = []
    for i in range(len(X_val)):
        val_features.append(InputFeatures(input_ids=val_encodings['input_ids'][i],
                                          attention_mask=val_encodings['attention_mask'][i],
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
    all_labels = list(all_labels)
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


def train_and_evaluate(all_labels, train_dataset, val_dataset, label_map):
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
    print('Training Roberta pre-trained model...')
    print('Classifying...')

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

    return trainer, accuracy, report


if __name__ == "__main__":
    # Get data generated previously to spare computing effort
    X_train, y_train, X_dev, y_dev, X_test, y_test, all_labels = recover_data()

    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_dev = X_dev[:100]
    y_dev = y_dev[:100]


    # DEV SET

    train_dataset, val_dataset, tokenizer, label_map = prepare_datasets(X_train, y_train, X_dev, y_dev, all_labels)
    train_and_evaluate(all_labels, train_dataset, val_dataset, label_map)

    # model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(all_labels))
    #
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     num_train_epochs=3,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    # )
    #
    # trainer = train_model(model, train_dataset, val_dataset, training_args)
    #
    # accuracy, actual_labels, predicted_labels, actual_labels_numeric = evaluate_model(trainer, val_dataset, all_labels,
    #                                                                                   label_map)
    # unique_labels = set(actual_labels)
    # num_classes = len(unique_labels)
    #
    # print("Accuracy:", accuracy)  # For the first 100 Accuracy: 0.65 no features
    # # 1000 acc 0.695
    #
    # if len(unique_labels) != num_classes:
    #     print("Number of unique labels:", len(unique_labels))
    #     print("Expected number of classes:", num_classes)
    #     print("Actual labels:", unique_labels)
    #     print("Predicted labels:", predicted_labels)
    #     print("All labels:", all_labels)
    #
    # report = generate_classification_report(actual_labels_numeric, [label_map[label] for label in predicted_labels],
    #                                         all_labels)

    # TEST SET
    train_dataset, test_dataset, tokenizer, label_map = prepare_datasets(X_train, y_train, X_test, y_test, all_labels)
    train_and_evaluate(all_labels, train_dataset, test_dataset, label_map)
