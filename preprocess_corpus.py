# Sandra Sánchez Páez
# ArgMin Modulprojekt

import os
from pathlib import Path
import re
from tokenize import tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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


def label_non_argumentative(all_annotations_file, all_text_file, output_file):
    # Read all annotations into a dictionary
    annotations = {}
    with open(all_annotations_file, 'r') as ann_file:
        for line in ann_file:
            parts = line.strip().split('\t')
            filename = parts[0]
            label = parts[1]
            if filename not in annotations:
                annotations[filename] = {}
            start_index, end_index = map(int, parts[2].split(':'))
            annotations[filename][(start_index, end_index)] = label

    # Open the all_text_file and read paragraphs
    with open(all_text_file, 'r') as text_file, open(output_file, 'w') as output:
        current_article = None
        current_paragraph = ""
        for line in text_file:
            line = line.strip()
            if line.startswith("essay"):
                if current_article:
                    # Process the current paragraph
                    labels = annotations.get(current_article, {})
                    labeled = False
                    for start_end, label in labels.items():
                        if current_paragraph.find(line) != -1:
                            output.write(f"{current_article}\t{label}\t{current_paragraph}\n")
                            labeled = True
                            break
                    if not labeled:
                        output.write(f"{current_article}\tnon-argumentative\t{current_paragraph}\n")
                current_article = line
                current_paragraph = ""
            else:
                current_paragraph += line + " "


def group_text_spans_by_article(corpus_file, annotations_file):
    # Read the corpus file
    with open(corpus_file, 'r') as f:
        corpus_content = f.read()
    # Read annotations into a DataFrame
    annotations_df = pd.read_csv(annotations_file, sep='\t', header=None, names=['File', 'Label', 'Text'])

    # Create a dictionary to store text spans by article
    articles_text_spans = {}
    for _, row in annotations_df.iterrows():
        article_id = row['File'].split('.')[0]  # Extract article ID from the filename
        if article_id not in articles_text_spans:
            articles_text_spans[article_id] = []
        articles_text_spans[article_id].append(row['Text'])

    # Concatenate text spans for each article
    articles_concatenated_text = {}
    for article_id, text_spans in articles_text_spans.items():
        articles_concatenated_text[article_id] = '\n'.join(text_spans)

    return articles_concatenated_text


def label_text_spans_with_annotations(articles_file, annotations_file):
    # Read all articles from the corpus file
    with open(articles_file, 'r') as f:
        articles_content = f.read()

    # Read annotations into a DataFrame, skipping the header row
    annotations_df = pd.read_csv(annotations_file, sep='\t', header=None, names=['File', 'Label', 'Text'])

    text_and_labels = []

    # Iterate through each article
    for article in articles_content.split('Article '):
        if article.strip():  # Skip empty articles (if any)
            article_lines = article.split('\n')
            article_name = article_lines[0].strip()  # Extract article name
            article_text = '\n'.join(article_lines[1:]).strip()  # Extract article text

            # Check if any annotations exist for this article
            annotations_for_article = annotations_df[annotations_df['File'] == article_name]

            if not annotations_for_article.empty:
                # Annotations found for this article
                for _, row in annotations_for_article.iterrows():
                    text_span = row['Text']
                    label = row['Label']
                    text_and_labels.append(f"{text_span}\t{label}")
            else:
                # No annotations found, label the entire article as 'non-argumentative'
                text_and_labels.append(f"{article_text}\tnon-argumentative")

    return text_and_labels


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


def label_articles(all_articles_file, all_annotations_file):
    articles = {}
    with open(all_articles_file, 'r') as file:
        current_article = None
        current_text = ""
        for line in file:
            line = line.strip()
            if line.startswith("Article"):
                if current_article:
                    articles[current_article] = current_text.strip()
                current_article = line
                current_text = ""
            else:
                current_text += line + "\n"
        if current_article:
            articles[current_article] = current_text.strip()

    annotations = read_annotations(all_annotations_file)

    labeled_articles = {}
    for article, text in articles.items():
        labeled_paragraphs = process_article(text, annotations.get(article, {}))
        labeled_articles[article] = labeled_paragraphs

    return labeled_articles


def read_annotations(annotations_file):
    annotations = {}
    with open(annotations_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            filename = parts[0]
            label = parts[1]
            annotations[(filename, label)] = parts[2]
    return annotations


def label_articles(all_articles_file, all_annotations_file):
    articles = {}
    with open(all_articles_file, 'r') as file:
        current_article = None
        current_text = ""
        for line in file:
            line = line.strip()
            if line.startswith("Article"):
                if current_article:
                    annotations = read_annotations(all_annotations_file)
                    processed_text = process_article(current_text, annotations.get(current_article, {}))
                    articles[current_article] = processed_text
                current_article = line
                current_text = ""
            else:
                current_text += " " + line
        if current_article:
            annotations = read_annotations(all_annotations_file)
            processed_text = process_article(current_text, annotations.get(current_article, {}))
            articles[current_article] = processed_text
    return articles


def process_article(article_text, annotations):
    processed_text = []
    current_paragraph = ""
    current_label = None
    for line in article_text.split('\n'):
        if line.strip() == "":
            if current_paragraph.strip() != "":
                if current_label:
                    processed_text.append((current_paragraph.strip(), current_label))
                else:
                    processed_text.append((current_paragraph.strip(), 'non_argumentative'))
                current_paragraph = ""
                current_label = None
        else:
            label = None
            for start_end, label_value in annotations.items():
                start, end = start_end
                if re.search(r'\b{}\b'.format(label_value), line, re.IGNORECASE):
                    label = label_value
                    break
            if label:
                if current_paragraph.strip() != "":
                    if current_label:
                        processed_text.append((current_paragraph.strip(), current_label))
                    else:
                        processed_text.append((current_paragraph.strip(), 'non_argumentative'))
                    current_paragraph = ""
                current_label = label
                processed_text.append((line.strip(), current_label))
            else:
                current_paragraph += " " + line.strip()
    if current_paragraph.strip() != "":
        if current_label:
            processed_text.append((current_paragraph.strip(), current_label))
        else:
            processed_text.append((current_paragraph.strip(), 'non_argumentative'))
    return processed_text


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
    # Create a dictionary to store labelled sentences
    labelled_sentences = {}

    # Iterate over each article
    for index, article_row in articles_dataframe.iterrows():
        article_id = article_row['id']
        article_text = article_row['text']

        # print(article_id)
        # print(article_text)
        # print('')

        # Initialize a list to store labelled fragments of text for this article
        labelled_fragments = []

        # Iterate over claims with the same id
        for _, claim_row in claims_n_premises_dataframe[claims_n_premises_dataframe['id'] == article_id].iterrows():
            claim_text = claim_row['text']
            label = claim_row['label']
            # print(f"Label '{label}': '{claim_text}'")

            # Check if the claim text is a part of the article text
            if claim_text in article_text:
                labelled_fragments.append((claim_text, label))

        # Split the article text based on labelled fragments
        start_index = 0
        for fragment, label in labelled_fragments:
            fragment_index = article_text.find(fragment, start_index)
            if fragment_index != -1:
                labelled_sentences[(article_id, start_index, fragment_index)] = 'non-argumentative'
                labelled_sentences[(article_id, fragment_index, fragment_index + len(fragment))] = label
                start_index = fragment_index + len(fragment)

        # Assign 'non-argumentative' label to the remaining part of the article
        if start_index < len(article_text):
            labelled_sentences[(article_id, start_index, len(article_text))] = 'non-argumentative'

        # print('_________________________________________')


    return labelled_sentences


# concatenate_txt_files(CORPUS, ALL_ARTICLES)
process_annotations(SEMANTIC_TYPES, 'all_sorted_annotated_texts.txt', CORPUS)

merge_txt_files(CORPUS)  # Returns 'merged_output.txt'
articles_df, claims_n_premises_df = transform_files_to_dataframes('merged_output.txt', 'all_sorted_annotated_texts.txt')
# print(articles_df[:10])
# print(claims_n_premises_df[:10])

labelled_sentences = get_labelled_sentences_from_data(articles_df, claims_n_premises_df)
print(labelled_sentences)







