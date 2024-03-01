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

from evaluate import generate_classification_report
from features import check_claim_verbs, extract_dependency_features_for_corpus, extract_ngram_features_for_corpus, \
    extract_topic_features


ROOT = Path(__file__).parent.resolve()
CORPUS = Path(f"{ROOT}/corpus/")
ALL_ARTICLES = Path(f"{ROOT}/all_articles.txt")
SEMANTIC_TYPES = Path(f"{ROOT}/essays_semantic_types.tsv")
ALL_ANNOTATIONS = Path(f"{ROOT}/all_sorted_annotated_texts.txt")


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


def merge_txt_files(directory):
    output_file = ALL_ARTICLES

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


def get_labelled_sentences_from_data(articles_dataframe, claims_n_premises_dataframe, preprocess=False):
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
                    unlabelled_fragment = article_text[start_index:fragment_index].strip()  # Strip whitespace
                    # Check if the unlabelled fragment is not empty
                    if unlabelled_fragment:
                        if preprocess:
                            preprocessed_fragment = preprocess_text(unlabelled_fragment)
                        else:
                            preprocessed_fragment = unlabelled_fragment
                        # Check if the preprocessed fragment is not empty
                        if preprocessed_fragment:
                            texts.append(preprocessed_fragment)
                            labels.append('non-argumentative')

                # Store the labelled fragment
                if preprocess:
                    preprocessed_claim = preprocess_text(claim_text)
                else:
                    preprocessed_claim = claim_text
                # Check if the preprocessed claim is not empty
                if preprocessed_claim:
                    texts.append(preprocessed_claim)
                    labels.append(label)

                # Update indices for the next iteration
                start_index = fragment_index + len(claim_text)
                end_index = start_index

        # Store the remaining unlabelled fragment, if any
        if end_index < len(article_text):
            unlabelled_fragment = article_text[end_index:].strip()  # Strip whitespace
            # Check if the unlabelled fragment is not empty
            if unlabelled_fragment:
                if preprocess:
                    preprocessed_fragment = preprocess_text(unlabelled_fragment)
                else:
                    preprocessed_fragment = unlabelled_fragment
                # Check if the preprocessed fragment is not empty
                if preprocessed_fragment:
                    texts.append(preprocessed_fragment)
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


def encode_and_split_data(texts, labels, dev_size=0.1, test_size=0.1, random_state=42):
    # Encode labels for multilabel classification
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(labels)

    # Split the dataset into training, development, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(texts, encoded_labels, test_size=(dev_size + test_size), random_state=random_state)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(dev_size + test_size), random_state=random_state)

    return X_train, X_dev, X_test, y_train, y_dev, y_test, mlb


def create_term_document_matrix(X_train, y_train, X_dev, y_dev, include_additional_features=False):
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the training data and transform the data
    term_doc_matrix_train = vectorizer.fit_transform(X_train)
    # print(term_doc_matrix_train)
    term_doc_matrix_dev = vectorizer.transform(X_dev)
    print(term_doc_matrix_dev.shape)

    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_dev = np.array(y_dev)

    # Check vectorization parameters
    print("Vectorization Parameters:")
    print("Vectorizer Parameters:", vectorizer.get_params())

    # Check vocabulary size
    print("\nVocabulary Size:", len(vectorizer.vocabulary_))

    # Extract additional features if provided
    if include_additional_features:
        # 1) NGRAMS
        ngram_matrix_train, _ = extract_ngram_features_for_corpus(X_train, vectorizer)
        ngram_matrix_dev, _ = extract_ngram_features_for_corpus(X_dev, vectorizer)
        print("term_doc_matrix_train shape:", term_doc_matrix_train.shape)
        print("term_doc_matrix_dev shape:", term_doc_matrix_dev.shape)
        print("ngram_matrix_train shape:", ngram_matrix_train.shape)
        print("ngram_matrix_dev shape:", ngram_matrix_dev.shape)

        # Concatenate additional n-gram features with the term-document matrices
        term_doc_matrix_train = np.hstack((term_doc_matrix_train.toarray(), ngram_matrix_train.toarray()))
        print("Matrix_train shape after concatenating ngrams:", term_doc_matrix_train.shape)
        term_doc_matrix_dev = np.hstack((term_doc_matrix_dev.toarray(), ngram_matrix_dev.toarray()))
        print("Matrix_dev shape after concatenating ngrams:", term_doc_matrix_dev.shape)

        # 2) TOPIC MODELLING
        # Extract topic modelling features for the corpus
        topic_words_train, topic_distributions_train = extract_topic_features(X_train, vectorizer)
        # print(topic_distributions_train)
        topic_features_dev, topic_distributions_dev = extract_topic_features(X_dev, vectorizer)
        # print(topic_distributions_dev)

        # Encode features
        encoded_topic_features_train = np.array(topic_distributions_train)
        encoded_topic_features_dev = np.array(topic_distributions_dev)
        print("Encoded topic features train shape:", encoded_topic_features_train.shape)
        print("Encoded topic features dev shape:", encoded_topic_features_dev.shape)

        # Concatenate TM features with the term-document matrices
        term_doc_matrix_train = np.hstack((term_doc_matrix_train, encoded_topic_features_train))
        print("Matrix_train shape after concatenating TM:", term_doc_matrix_train.shape)
        term_doc_matrix_dev = np.hstack((term_doc_matrix_dev, encoded_topic_features_dev))
        print("Matrix_dev shape after concatenating TM:", term_doc_matrix_dev.shape)

        # 3) DEPENDENCIES
        # Extract dependency features for the corpus
        dependency_matrix_train = extract_dependency_features_for_corpus(X_train)
        print("dependency_matrix_train shape:", dependency_matrix_train.shape)
        # Filter out empty lists and ensure all lists have consistent length
        # dependency_matrix_train = [d for d in dependency_matrix_train if d and len(d) > 0]

        dependency_matrix_dev = extract_dependency_features_for_corpus(X_dev, dependency_matrix_train)
        print("dependency_matrix_dev shape:", dependency_matrix_dev.shape)
        # dependency_matrix_dev = [d for d in dependency_matrix_dev if d and len(d) > 0]

        # # Convert dependency matrices from lists to NumPy arrays
        dependency_matrix_train = np.array(dependency_matrix_train)
        # print("dependency_matrix_train shape:", dependency_matrix_train.shape)
        dependency_matrix_dev = np.array(dependency_matrix_dev)
        # print("dependency_matrix_dev shape:", dependency_matrix_dev.shape)

        # Concatenate dependency features with the term-document matrices
        term_doc_matrix_train = np.hstack((term_doc_matrix_train, dependency_matrix_train))
        print("Matrix_train shape after concatenating dependencies:", term_doc_matrix_train.shape)
        term_doc_matrix_dev = np.hstack((term_doc_matrix_dev, dependency_matrix_dev))
        print("Matrix_dev shape after concatenating dependencies:", term_doc_matrix_dev.shape)

        # 4) CLAIM VERBS
        # Check for claim-related verbs
        claim_train = [check_claim_verbs(text) for text in X_train]
        claim_dev = [check_claim_verbs(text) for text in X_dev]
        # Convert to arrays
        claim_train = np.array(claim_train)
        # Reshape claim_train to a column vector
        claim_train = claim_train.reshape(-1, 1)
        print("claim_matrix_train shape:", claim_train.shape)
        claim_dev = np.array(claim_dev)
        print("claim_matrix_dev shape:", claim_dev.shape)
        # Reshape claim_train to a column vector
        claim_dev = claim_dev.reshape(-1, 1)
        # Concatenate
        term_doc_matrix_train = np.hstack((term_doc_matrix_train, claim_train))
        print("Matrix_train shape after concatenating claim-related features:", term_doc_matrix_train.shape)
        term_doc_matrix_dev = np.hstack((term_doc_matrix_dev, claim_dev))
        print("Matrix_dev shape after concatenating claim-related features:", term_doc_matrix_dev.shape)

    print('Matrices with all features fitted')
    return term_doc_matrix_train, y_train, term_doc_matrix_dev, y_dev


def save_matrices(X_train_matrix, y_train, X_dev_matrix, y_dev):
    save_data(X_train_matrix, 'X_train.pkl')
    save_data(X_dev_matrix, 'X_dev.pkl')
    save_data(y_train, 'y_train.pkl')
    save_data(y_dev, 'y_dev.pkl')


def load_matrices():
    X_train = load_data('X_train.pkl')
    X_dev = load_data('X_dev.pkl')
    y_train = load_data('y_train.pkl')
    y_dev = load_data('y_dev.pkl')

    return X_train, X_dev, y_train, y_dev


if __name__ == '__main__':
    # concatenate_txt_files(CORPUS, ALL_ARTICLES)
    process_annotations(SEMANTIC_TYPES, ALL_ANNOTATIONS, CORPUS)
    merge_txt_files(CORPUS)  # Returns 'all_articles.txt'
    articles_df, claims_n_premises_df = transform_files_to_dataframes(ALL_ARTICLES, ALL_ANNOTATIONS)

    # Get and preprocess labelled texts
    texts, labels = get_labelled_sentences_from_data(articles_df, claims_n_premises_df)
    print(f"There are {len(texts)} texts") # 12570
    print(f"There are {len(labels)} labels")
    save_data(texts, 'texts.pkl')
    save_data(labels, 'labels.pkl')

    # preprocessed_texts, preprocessed_labels = get_labelled_sentences_from_data(articles_df, claims_n_premises_df, preprocess=True)
    # print(f"There are {len(preprocessed_texts)} preprocessed texts") # 9712 after empty texts removed
    # print(f"There are {len(preprocessed_labels)} preprocessed labels")
    # save_data(preprocessed_texts, 'preprocessed_texts.pkl')
    # save_data(preprocessed_labels, 'preprocessed_labels.pkl')
    # labels = preprocessed_labels
    # texts = preprocessed_texts

    # Get unique labels
    unique_labels = set(labels)
    num_classes = len(unique_labels)
    print(f"The number of classes is {num_classes}")

    # Encode and split the data
    X_train, X_dev, X_test, y_train, y_dev, y_test, mlb = encode_and_split_data(texts, labels)
    print(len(X_train))
    print(len(y_train))
    print(len(X_dev))
    print(len(y_dev))
    print(len(X_test))
    print(len(y_test))

    print(X_test)
    print(y_test)


    # With features claculated within matrix function
    X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix, y_dev = create_term_document_matrix(X_train, y_train, X_dev, y_dev, include_additional_features=True)
    save_matrices(X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix, y_dev)
    # X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix, y_dev = load_matrices()

    # Initialize the RandomForestClassifier
    classifier = RandomForestClassifier()
    sleep(2)
    # Train the classifier
    classifier.fit(X_train_term_doc_matrix, y_train)
    sleep(2)
    # Predict on the development set
    y_dev_pred = classifier.predict(X_dev_term_doc_matrix)
    sleep(2)
    # Evaluate the classifier
    accuracy = accuracy_score(y_dev, y_dev_pred)
    print("Development Set Accuracy:", accuracy)  # 0.4789180588703262 no features, 0.48369132856006364 2 features, 0.4813046937151949 3 features
    # After removing empty texts: 0.36728395061728397 no features, 0.45823389021479716 features
    report = generate_classification_report(y_dev, y_dev_pred, unique_labels)



