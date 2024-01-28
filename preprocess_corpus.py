# Sandra Sánchez Páez
# ArgMin Modulprojekt

import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

CORPUS = Path(f"{ROOT}/corpus/")

ALL_ARTICLES = Path(f"{ROOT}/corpus/all_articles.txt")

ALL_ANNOTATIONS = Path(f"{ROOT}/essays_semantic_types.tsv")


def concatenate_txt_files(directory, output_file):
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
                    outfile.write(infile.read())
                    # Add a newline separator between files
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
        with open(output_file, 'w') as output:
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    filename = parts[0] + '.txt'
                    annotation_type = parts[1]
                    start_index, end_index = map(int, parts[2].split(':'))
                    # Locate text span in the file
                    text_span = locate_text_span(directory, filename, start_index, end_index)
                    if text_span is not None:
                        # Write annotation to output file
                        output.write(f"{filename}\t{annotation_type}\t{text_span}\n")
                    else:
                        print(f"Failed to locate text span for annotation in file '{filename}'.")


span = locate_text_span(CORPUS, 'essay001.txt', 503, 575)
if span is not None:
    print("Text span found:", span)

# Example usage:
tsv_file = "/path/to/annotations.tsv"
output_file = "/path/to/output_annotations.txt"
directory_path = "/path/to/your/text_files_directory"
process_annotations(ALL_ANNOTATIONS, 'all_annotated_texts.txt', CORPUS)
