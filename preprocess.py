# Sandra Sánchez Páez
# ArgMin Modulprojekt

import os
import string
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).parent.resolve()
CORPUS = Path(f"{ROOT}/corpus/")
ALL_ARTICLES = Path(f"{ROOT}/all_articles.txt")
SEMANTIC_TYPES = Path(f"{ROOT}/essays_semantic_types.tsv")
ALL_ANNOTATIONS = Path(f"{ROOT}/all_sorted_annotated_texts.txt")

import pandas as pd


def locate_text_span(directory, filename, start_index, end_index):
    """
    Locate and extract a text span from a file.
    Args:
    directory (str): The directory path where the file is located.
    filename (str): The name of the file to search within.
    start_index (int): The starting index of the text span.
    end_index (int): The ending index of the text span.
    Returns:
    str: The extracted text span from the specified file.
    Returns None if the directory or file does not exist.
    """
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
    """
    Process annotations from a (TSV) file and extract corresponding text spans from files.
    Args:
    tsv_file (str): The path to the TSV file containing annotations.
    output_file (str): The path to the output file where processed annotations will be saved.
    directory (str): The directory path where the files referenced in annotations are located.
    Returns: None
    """
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


def merge_txt_files(directory):
    """
    Merge text files (articles) from a directory (corpus) into a single file.
    Args:
    directory (str): The path to the directory containing text files to merge.
    Returns: None
    """
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
    """
    Transform text files into pandas DataFrames.
    Args:
    articles_file (str): Path to the file containing articles.
    annotations_file (str): Path to the file containing annotations.
    Returns:
    Two DataFrames, the first containing articles and the second containing claims and premises.
    """

    # Get text of all articles
    # Convert the data into a DataFrame
    articles_dataframe = pd.read_csv(articles_file, sep="\t", header=None, names=['id', 'text'])
    # Get all claims
    claims_n_premises_dataframe = pd.read_csv(annotations_file, sep="\t")
    claims_n_premises_dataframe.columns = ['id', 'label', 'text']

    return articles_dataframe, claims_n_premises_dataframe


def get_labelled_sentences_from_data(articles_dataframe, claims_n_premises_dataframe, preprocess=False):
    """
    Extract labelled sentences from articles and claims dataFrames.
    Args:
    articles_dataframe (pd.DataFrame): DataFrame containing articles.
    claims_n_premises_dataframe (pd.DataFrame): DataFrame containing claims and premises.
    preprocess (bool, optional): Flag indicating whether to preprocess text. Defaults to False.
    Returns:
    Two lists, the first containing texts and the second containing labels.
    """
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
            unlabelled_fragment = article_text[end_index:].strip()
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
    """
    Preprocess input text by tokenizing, removing punctuation,
    lowercase, removing stopwords, and lemmatizing.
    Args:
    text (str): Input text to be preprocessed.
    Returns:
    str: Preprocessed text.
    """
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


def preprocess_texts_and_labels(X_set, y_set):
    """Preprocess all texts and labels and filter empty ones out."""
    texts = []
    labels = []
    for text, label in zip(X_set, y_set):
        preprocessed_text = preprocess_text(text)
        if preprocessed_text:
            texts.append(preprocessed_text)
            labels.append(label)

    print('Number of preprocessed texts:', len(texts))
    print('Number of preprocessed labels:', len(labels))
    print('')

    return texts, labels
