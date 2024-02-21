# Sandra Sánchez Páez
# ArgMin Modulprojekt
from time import sleep

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import os
import pickle
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from tokenize import tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import RobertaForSequenceClassification

ROOT = Path(__file__).parent.resolve()
CORPUS = Path(f"{ROOT}/corpus/")
ALL_ARTICLES = Path(f"{ROOT}/all_articles.txt")
SEMANTIC_TYPES = Path(f"{ROOT}/essays_semantic_types.tsv")
ALL_ANNOTATIONS = Path(f"{ROOT}/all_sorted_annotated_texts")


def concatenate_txt_files(directory, output_file):
    """Write file with all articles texts and numbers"""
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Open the output file in append mode
    with open(output_file, 'w') as outfile:
        # Iterate over files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as infile:
                    # Read content of each file and write to the output file
                    article_number = filename.split('.')[0]
                    content = infile.read()
                    # Write article number and text content to the output file
                    outfile.write(f"Article {article_number}\n")
                    outfile.write(content)
                    # Add a newline separator between articles
                    outfile.write('\n')


def find_text_span(file_path, start_index, end_index):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' does not exist.")
        return None

    # Open the file and read its content
    with open(file_path, 'r') as file:
        content = file.read()
        text_span = content[start_index:end_index]
        return text_span


def locate_text_span(directory, filename, start_index, end_index):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return None

    # Check if the file exists
    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
        print(f"File '{filename}' does not exist in directory '{directory}'.")
        return None

    # Open the file and read its content
    with open(file_path, 'r') as file:
        content = file.read()
        text_span = content[start_index:end_index]
        return text_span


def process_annotations(tsv_file, output_file, directory):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Open the TSV file and read annotations
    with open(tsv_file, 'r') as tsv:
        lines = tsv.readlines()[1:]  # Skip the first line (assuming it contains column names)
        annotations = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                filename = parts[0] + '.txt'
                annotation_type = parts[1]
                start_index, end_index = map(int, parts[2].split(':'))
                # Locate text span in the file
                text_span = locate_text_span(directory, filename, start_index, end_index)
                if text_span is not None:
                    # Store the annotation along with article ID and start index
                    annotations.append((filename, annotation_type, text_span, start_index))
                else:
                    print(f"Failed to locate text span for annotation in file '{filename}'.")

    # Sort the annotations by article ID and start index
    annotations.sort(key=lambda x: (x[0], x[3]))

    # Write sorted annotations to the output file
    with open(output_file, 'w') as output:
        for annotation in annotations:
            filename, annotation_type, text_span, _ = annotation
            filename_parts = filename.split('.')
            filename = filename_parts[0]
            output.write(f"{filename}\t{annotation_type}\t{text_span}\n")


def process_article(article_text, annotations):
    processed_paragraphs = []
    for paragraph in article_text.split('\n'):
        if paragraph.strip() == "":
            continue
        label = 'non_argumentative'
        for start_end, label_value in annotations.items():
            start, end = start_end
            if paragraph.find(label_value) != -1:
                label = label_value
                break
        processed_paragraphs.append((paragraph, label))
    return processed_paragraphs


def read_annotations(annotations_file):
    annotations = {}
    with open(annotations_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            filename = parts[0]
            label = parts[1]
            annotations[(filename, label)] = parts[2]
    return annotations


def read_annotations(annotations_file):
    annotations = {}
    with open(annotations_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            filename = parts[0]
            label = parts[1]
            annotations[(filename, label)] = parts[2]
    return annotations


def merge_txt_files(directory):
    output_file = "merged_output.txt"

    # Get a sorted list of filenames in the directory
    sorted_filenames = sorted(os.listdir(directory))

    with open(output_file, "w") as merged_file:
        for filename in sorted_filenames:
            if filename.endswith(".txt"):
                with open(os.path.join(directory, filename), "r") as txt_file:
                    title = filename.split('.')[0]  # Extract title from filename
                    # print(title)
                    content = ' '.join(txt_file.read().split())  # Remove consecutive spaces
                    merged_file.write(f"{title}\t{content}\n")  # Use tab as separator

    print(f"Merged content written to {output_file}")


def transform_files_to_dataframes(articles_file, annotations_file):
    # Get text of all articles
    # Convert the data into a DataFrame
    articles_dataframe = pd.read_csv(articles_file, sep="\t", header=None, names=['id', 'text'])
    # Get all claims
    claims_n_premises_dataframe = pd.read_csv(annotations_file, sep="\t")
    claims_n_premises_dataframe.columns = ['id', 'label', 'text']

    return articles_dataframe, claims_n_premises_dataframe


def get_labelled_sentences_from_data(articles_dataframe, claims_n_premises_dataframe):
    # Create empty lists to store texts and labels
    texts = []
    labels = []

    # Iterate over each article
    for index, article_row in articles_dataframe.iterrows():
        article_id = article_row['id']
        article_text = article_row['text']

        # Initialize start and end indices for tracking fragments
        start_index = 0
        end_index = 0

        # Iterate over claims with the same id
        for _, claim_row in claims_n_premises_dataframe[claims_n_premises_dataframe['id'] == article_id].iterrows():
            claim_text = claim_row['text']
            label = claim_row['label']

            # Find the position of the claim text in the article text
            fragment_index = article_text.find(claim_text, start_index)

            # If the claim text is found in the article text
            if fragment_index != -1:
                # Store the unlabelled fragment before the labelled fragment
                if start_index < fragment_index:
                    unlabelled_fragment = article_text[start_index:fragment_index]
                    texts.append(unlabelled_fragment)
                    labels.append('non-argumentative')

                # Store the labelled fragment
                texts.append(claim_text)
                labels.append(label)

                # Update indices for the next iteration
                start_index = fragment_index + len(claim_text)
                end_index = start_index

        # Store the remaining unlabelled fragment, if any
        if end_index < len(article_text):
            unlabelled_fragment = article_text[end_index:]
            texts.append(unlabelled_fragment)
            labels.append('non-argumentative')

    return texts, labels


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Lowercase all words
    tokens = [word.lower() for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def preprocess_text_fragments(all_texts, filename):
    X_preprocessed = []
    for text in tqdm(all_texts):
        preprocessed_text = preprocess_text(text)
        # print(text)
        # print(preprocessed_text)
        X_preprocessed.append(preprocessed_text)
    #     X_preprocessed.append(' '.join(preprocessed_text))
    print('Saving data...')
    save_data(X_preprocessed, filename)
    print(f"The number of preprocessed texts is {len(X_preprocessed)}.")
    print(f"The preprocessed sentences have been saved and will be available as {filename}")

    return X_preprocessed


def save_data(data: any, filename: any) -> None:
    """Save data into file_name (.pkl file)to save time."""
    with open(filename, mode="wb") as file:
        pickle.dump(data, file)
        print(f'Data saved in {filename}')


def load_data(filename: any) -> any:
    """Load pre-saved data."""
    with open(filename, "rb") as file:
        output = pickle.load(file)
    print(f'Loading  data  pre-saved as {filename}...')
    return output


def encode_and_split_data(texts, labels, test_size=0.2, dev_size=0.1, random_state=42):
    # Encode labels for multilabel classification
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=test_size, random_state=random_state)

    # Split the training set further into training and development sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=dev_size/(1-test_size), random_state=random_state)

    return X_train, X_dev, X_test, y_train, y_dev, y_test, mlb


def create_term_document_matrix(X_train, y_train, X_dev, y_dev, additional_features=None):
    # Initialize a CountVectorizer or TfidfVectorizer
    vectorizer = TfidfVectorizer()  # You can change to CountVectorizer if you want simple word counts

    # Fit the vectorizer to the data and transform the data
    term_doc_matrix = vectorizer.fit_transform(X_train)
    # print(term_doc_matrix)
    y_train = np.array(y_train)
    X_dev = vectorizer.transform(X_dev)
    print(X_dev.shape)
    y_dev = np.array(y_dev)

    # Check vectorization parameters
    print("Vectorization Parameters:")
    print("Vectorizer Parameters:", vectorizer.get_params())

    # Check vocabulary size
    print("\nVocabulary Size:", len(vectorizer.vocabulary_))

    # Extract additional features if provided
    if additional_features:
        # Convert additional features to numpy array
        additional_features_array = np.array(additional_features)
        print("\nAdditional Features Shape:", additional_features_array.shape)

        # Concatenate additional features with the term-document matrix
        term_doc_matrix = np.hstack((term_doc_matrix.toarray(), additional_features_array))

    print('Matrix with all features fitted')
    return term_doc_matrix, y_train, X_dev, y_dev


# concatenate_txt_files(CORPUS, ALL_ARTICLES)
process_annotations(SEMANTIC_TYPES, 'all_sorted_annotated_texts.txt', CORPUS)

merge_txt_files(CORPUS)  # Returns 'merged_output.txt'
articles_df, claims_n_premises_df = transform_files_to_dataframes('merged_output.txt', 'all_sorted_annotated_texts.txt')
# print(articles_df[:10])
# print(claims_n_premises_df[:10])

texts, labels = get_labelled_sentences_from_data(articles_df, claims_n_premises_df)
# Get unique labels
unique_labels = set(labels)
num_classes = len(unique_labels)
print(f"The number of classes is {num_classes}")
# print(texts)
# print(labels)
# print(len(labels))  # 12570
# preprocessed_texts = preprocess_text_fragments(texts, 'preprocessed_texts.pkl')  # The number of preprocessed sentences is 12570.
# print(preprocessed_texts)
preprocessed_texts = load_data('preprocessed_texts.pkl')

# Encode and split the data
X_train, X_dev, X_test, y_train, y_dev, y_test, mlb = encode_and_split_data(preprocessed_texts, labels)
print(len(X_train))
print(len(y_train))
print(len(X_dev))
print(len(y_dev))

# Assuming X_train, X_dev, and X_test are your text data, and additional_features_train,
# additional_features_dev, and additional_features_test are additional features relevant
# for claims and premises
# You need to replace them with your actual data
X_train_term_doc_matrix, y_train, X_dev, y_dev = create_term_document_matrix(X_train, y_train, X_dev, y_dev)
# X_train_term_doc_matrix = create_term_document_matrix(X_train, additional_features=additional_features_train)

# save_data(X_train_term_doc_matrix, 'ov_X_train.pkl')
# save_data(X_dev, 'ov_X_dev.pkl')
# save_data(y_train, 'ov_y_train.pkl')
# save_data(y_dev, 'ov_y_dev.pkl')

load_data('ov_X_train.pkl')
load_data('ov_X_dev.pkl')
load_data('ov_y_train.pkl')
load_data('ov_y_dev.pkl')

# Initialize the RandomForestClassifier
classifier = RandomForestClassifier()
sleep(3)
# Train the classifier
classifier.fit(X_train_term_doc_matrix, y_train)
sleep(3)
# Predict on the development set
y_dev_pred = classifier.predict(X_dev)
sleep(3)
# Evaluate the classifier
accuracy = accuracy_score(y_dev, y_dev_pred)
print("Development Set Accuracy:", accuracy)  # 0.4789180588703262

# TRANSFORMERS
# Step 1: Prepare Data
# Assuming you have your dataset loaded into X (text data) and y (semantic types)

# Encode labels into numerical format
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into training, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Tokenize the texts (you can replace this with your own tokenizer if needed)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Convert labels to tensors
train_labels_tensor = torch.tensor(train_labels)
val_labels_tensor = torch.tensor(val_labels)
test_labels_tensor = torch.tensor(test_labels)

# Define the datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings["input_ids"]),
    torch.tensor(train_encodings["attention_mask"]),
    train_labels_tensor
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings["input_ids"]),
    torch.tensor(val_encodings["attention_mask"]),
    val_labels_tensor
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings["input_ids"]),
    torch.tensor(test_encodings["attention_mask"]),
    test_labels_tensor
)


# Step 2: Tokenization
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# print('...Downloaded Robertatokenizer')

# # Tokenize text data
# tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# print('...Tokenized data')
# print('Saving data...')
# save_data(tokenized_data, 'tokenized_data.pkl')
# print(f"The preprocessed data has been saved and will be available as 'tokenized_data.pkl'")

# tokenized_data = load_data('tokenized_data.pkl')
# sleep(2)
#
# # Use MultiLabelBinarizer to convert multilabel strings to binary encoded labels
# label_binarizer = MultiLabelBinarizer()
# binary_labels = label_binarizer.fit_transform(labels)
#
# # Split data into train and validation sets
# train_inputs, val_inputs, train_labels, val_labels = train_test_split(
#     tokenized_data, binary_labels, test_size=0.2
# ) #todo: should take preprocessed with roberta tokenizer but needs to be reformatted
# print('Did train-test split')
# sleep(3)
#
# # Convert labels to tensors
# train_labels_tensor = torch.tensor(train_labels)
# val_labels_tensor = torch.tensor(val_labels)

# # Define the datasets
# train_dataset = torch.utils.data.TensorDataset(
#     train_inputs["input_ids"], train_inputs["attention_mask"], train_labels_tensor
# )
# val_dataset = torch.utils.data.TensorDataset(
#     val_inputs["input_ids"], val_inputs["attention_mask"], val_labels_tensor
# )
#
# # Make sure they contain the required input fields: input_ids, attention_mask, token_type_ids, labels
# # Print a sample to verify the structure
# print("Training Dataset Samples:")
# for sample in train_inputs[:5]:
#     print(sample)
#
# print("\nTraining  Labels Samples:")
# for label in binary_labels[:5]:
#     print(label)

# Step 3: Model Architecture
# model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_classes)
# print('...Loaded Roberta model')
# # Save the model to disk
# model.save_pretrained("roberta_sequence_classification_model")

# Load the model from the saved directory
model = RobertaForSequenceClassification.from_pretrained("roberta_sequence_classification_model")
print('...Loaded Roberta model')
sleep(2)
# Step 4: Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
sleep(1)
trainer.train()
print('...Trained Roberta model')
sleep(2)

# Step 5: Evaluation (optional)
eval_results = trainer.evaluate(eval_dataset=val_dataset)
print('...Evaluated Roberta model')

# Step 6: Fine-Tuning (optional)
# Adjust hyperparameters and repeat steps 4 and 5 to fine-tune the model further











