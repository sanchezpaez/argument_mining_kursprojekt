# Sandra Sánchez Páez
# ArgMin Modulprojekt
import pickle
from time import sleep

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from evaluate import generate_classification_report
from features import check_claim_verbs, extract_dependency_features_for_corpus, extract_ngram_features_for_corpus, \
    extract_topic_features
from preprocess import process_annotations, merge_txt_files, transform_files_to_dataframes, \
    get_labelled_sentences_from_data, CORPUS, SEMANTIC_TYPES, ALL_ANNOTATIONS, ALL_ARTICLES, preprocess_texts_and_labels


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


def split_data(texts, labels, dev_size=0.1, test_size=0.1, random_state=42):
    # Split the dataset into training, development, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=(dev_size + test_size),
                                                        random_state=random_state)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (dev_size + test_size),
                                                    random_state=random_state)

    print('Number of items in X_train:', len(X_train))  # 10056
    print('Number of items in y_train:', len(y_train))  # 10056
    print('Number of items in X_dev:', len(X_dev))  # 1257
    print('Number of items in y_dev:', len(y_dev))  # 1257
    print('Number of items in X_test:', len(X_test))  # 1257
    print('Number of items in y_test:', len(y_test))  # 1257

    save_data(X_train, 'X_train.pkl')
    save_data(y_train, 'y_train.pkl')
    save_data(X_dev, 'X_dev.pkl')
    save_data(y_dev, 'y_dev.pkl')
    save_data(X_test, 'X_test.pkl')
    save_data(y_test, 'y_test.pkl')
    print('')

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def encode_data(y_train, y_dev, y_test):
    # # Encode labels for multilabel classification
    # mlb = MultiLabelBinarizer()
    # encoded_labels_train = mlb.fit_transform(y_train)
    # encoded_labels_dev = mlb.transform(y_dev)
    # encoded_labels_test = mlb.transform(y_test)
    # # print(mlb.classes_)
    #
    # return encoded_labels_train, encoded_labels_dev, encoded_labels_test, mlb
    # Encode labels for multiclass classification
    label_encoder = LabelEncoder()
    encoded_labels_train = label_encoder.fit_transform(y_train)
    encoded_labels_dev = label_encoder.transform(y_dev)
    encoded_labels_test = label_encoder.transform(y_test)

    # Optionally, you can also get the classes if needed
    classes = label_encoder.classes_

    return encoded_labels_train, encoded_labels_dev, encoded_labels_test, classes


def create_term_document_matrix(X_train, y_train, X_dev, y_dev, include_additional_features=False):
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the training data and transform the data
    term_doc_matrix_train = vectorizer.fit_transform(X_train)
    # print(term_doc_matrix_train)
    term_doc_matrix_dev = vectorizer.transform(X_dev)

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
        print('Extracting dependency features...')
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
    save_data(X_train_matrix, 'X_train_matrix.pkl')
    save_data(X_dev_matrix, 'X_dev_matrix.pkl')
    save_data(y_train, 'y_train_matrix.pkl')
    save_data(y_dev, 'y_dev_matrix.pkl')


def load_matrices():
    X_train = load_data('X_train.pkl')
    X_dev = load_data('X_dev.pkl')
    y_train = load_data('y_train.pkl')
    y_dev = load_data('y_dev.pkl')

    return X_train, X_dev, y_train, y_dev


def reformat_corpus(directory, annotations_file, annotated_texts_file, article_file):
    process_annotations(annotations_file, annotated_texts_file, directory)
    merge_txt_files(directory)  # Returns 'all_articles.txt'
    articles_df, claims_n_premises_df = transform_files_to_dataframes(article_file, annotated_texts_file)

    print('Reformatting corpus...')

    return articles_df, claims_n_premises_df


def get_texts_and_labels(dataframe_articles, dataframe_arguments, preprocess=False):
    print('Processing data...')
    texts, labels = get_labelled_sentences_from_data(dataframe_articles,
                                                     dataframe_arguments)
    print("Number of texts:", len(texts))  # 12570
    print("Number of labels:", len(labels))  # 12570

    save_data(texts, 'texts.pkl')
    save_data(labels, 'labels.pkl')

    if preprocess:
        preprocessed_texts, preprocessed_labels = get_labelled_sentences_from_data(dataframe_articles, dataframe_arguments,
                                                                                   preprocess=True)
        print("Number of preprocessed texts:", len(texts))  # 9712 after empty texts removed
        print("Number of preprocessed labels:", len(labels))  # 9712 after empty texts removed
        save_data(preprocessed_texts, 'preprocessed_texts.pkl')
        save_data(preprocessed_labels, 'preprocessed_labels.pkl')
        labels = preprocessed_labels
        texts = preprocessed_texts

    unique_labels = get_unique_labels(labels)

    return texts, labels, unique_labels


def get_unique_labels(labels):
    unique_labels = set(labels)
    save_data(unique_labels, 'unique_labels.pkl')
    num_classes = len(unique_labels)
    print('Number of classes:', num_classes)  # 10

    return unique_labels


def train_baseline(X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix):
    # Initialize the RandomForestClassifier
    classifier = RandomForestClassifier()
    # Train the classifier
    classifier.fit(X_train_term_doc_matrix, y_train)
    print("Shape of X_train_term_doc_matrix:", X_train_term_doc_matrix.shape)
    print("Shape of y_train:", y_train.shape)

    # Predict on the development set
    y_dev_pred = classifier.predict(X_dev_term_doc_matrix)
    print("Shape of X_dev_term_doc_matrix:", X_dev_term_doc_matrix.shape)

    return y_dev_pred


def evaluate_baseline(y_val, y_val_predictions, all_labels):
    accuracy = accuracy_score(y_val, y_val_predictions)
    print("Accuracy score:", accuracy)

    # 0.4789180588703262 no features, 0.48369132856006364 2 features, 0.4813046937151949 3 features
    # No preprocess yes features: 0.4606205250596659, 0.4598249801113763 dev set
    # No preprocess no features: 0.4964200477326969 dev set
    # yes preprocess no features: 0.3779608650875386 dev set
    # yes preprocess yes features: 0.3367662203913491 dev set

    report = generate_classification_report(y_val, y_val_predictions, all_labels)

    return accuracy, report


def fit_classify_evaluate(dataset):
    if dataset == 'dev':
        # Fit data/Extract features
        X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix, y_dev = create_term_document_matrix(
            X_train_preprocessed,
            encoded_y_train,
            X_dev_preprocessed,
            encoded_y_dev,
            include_additional_features=True
        )
        save_matrices(X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix, y_dev)
        print('Number of items in y_sets')
        print(len(y_train))
        print(len(y_dev))

        # X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix, y_dev = load_matrices()

        # Train & Classify
        y_dev_pred = train_baseline(X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix)

        # Evaluate the classifier
        print("Development Set")
        accuracy_score, classification_report = evaluate_baseline(y_dev, y_dev_pred, unique_labels)
        save_data(accuracy_score, 'accuracy_dev_rfc.pkl')
        save_data(classification_report, 'classification_report_dev_rfc.pkl')

    elif dataset == 'test':
        # Fit data/Extract features
        X_train_term_doc_matrix, y_train, X_test_term_doc_matrix, y_test = create_term_document_matrix(
            X_train_preprocessed,
            encoded_y_train,
            X_test_preprocessed,
            encoded_y_test,
            include_additional_features=True
        )
        save_matrices(X_train_term_doc_matrix, y_train, X_test_term_doc_matrix, y_test)
        # X_train_term_doc_matrix, y_train, X_dev_term_doc_matrix, y_dev = load_matrices()

        # Train & Classify
        y_test_pred = train_baseline(X_train_term_doc_matrix, y_train, X_test_term_doc_matrix)

        # Evaluate the classifier
        print("Test Set")
        accuracy_score, classification_report = evaluate_baseline(y_test, y_test_pred, unique_labels)
        save_data(accuracy_score, 'accuracy_test_rfc.pkl')
        save_data(classification_report, 'classification_report_test_rfc.pkl')


if __name__ == '__main__':
    # # Reformat corpus
    # articles_df, claims_n_premises_df = reformat_corpus(
    #     directory=CORPUS, annotations_file=SEMANTIC_TYPES,
    #     annotated_texts_file=ALL_ANNOTATIONS, article_file=ALL_ARTICLES
    # )
    #
    # # Get and preprocess labelled texts
    # texts, labels, unique_labels = get_texts_and_labels(
    #     articles_df,
    #     claims_n_premises_df,
    #     preprocess=False # NO PREPROCESS IF WE SAVE DATA FOR ROBERTA!!
    # )
    #
    # # Encode and split the data
    # X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(texts, labels)
    # Load data
    X_train = load_data('X_train.pkl')
    y_train = load_data('y_train.pkl')
    X_dev = load_data('X_dev.pkl')
    y_dev = load_data('y_dev.pkl')
    X_test = load_data('X_test.pkl')
    y_test = load_data('y_test.pkl')
    unique_labels = load_data('unique_labels.pkl')
    print('Number of items in X_train:', len(X_train))  # 10056, 7787
    print('Number of items in y_train:', len(y_train))  # 10056, 7787
    print('Number of items in X_dev:', len(X_dev))  # 1257, 958
    print('Number of items in y_dev:', len(y_dev))  # 1257, 958
    print('Number of items in X_test:', len(X_test))  # 1257, 967
    print('Number of items in y_test:', len(y_test))  # 1257, 967

    print('')
    print('TRAINING SET')
    X_train_preprocessed, y_train_preprocessed = preprocess_texts_and_labels(X_train, y_train)
    print('DEVELOPMENT SET')
    X_dev_preprocessed, y_dev_preprocessed = preprocess_texts_and_labels(X_dev, y_dev)
    print('TEST SET')
    X_test_preprocessed, y_test_preprocessed = preprocess_texts_and_labels(X_test, y_test)

    # Preprocess?
    encoded_y_train, encoded_y_dev, encoded_y_test, classes = encode_data(y_train_preprocessed, y_dev_preprocessed, y_test_preprocessed)

    # Fit data/Extract features, Train & Classify, Evaluate
    #
    # DEV SET
    fit_classify_evaluate('dev')
    #
    # # TEST SET
    # fit_classify_evaluate('test')
