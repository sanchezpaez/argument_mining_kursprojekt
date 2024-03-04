from typing import Tuple, Any, List

import numpy as np
import scipy
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


def check_claim_verbs(sentence: list[str]) -> int:
    """
    Check if the given sentence contains any claim verbs.
    Args:
    sentence (list[str]): List of lemmas in the sentence.
    Returns:
    int: 1 if claim verb is found, 0 otherwise.
    """

    claim_verbs = ['argue', 'claim', 'emphasise', 'contend', 'maintain', 'assert', 'theorize', 'support the view that',
                   'deny', 'negate', 'refute', 'reject', 'challenge', 'strongly believe that', 'counter the view that',
                   'acknowledge', 'consider', 'discover', 'hypothesize', 'object', 'say', 'admit', 'assume', 'decide',
                   'doubt', 'imply', 'observe', 'show', 'agree', 'believe', 'demonstrate', 'emphasize', 'indicate',
                   'point out', 'state', 'allege', 'argument that']
    for lemma in sentence:
        if lemma in claim_verbs:
            return 1
    return 0


def extract_ngram_features_for_corpus(texts, vectorizer=None) -> Tuple[scipy.sparse.spmatrix, Any]:
    """
    Extract n-gram features from the given texts.
    Args:
    texts (list): List of texts to extract features from.
    vectorizer (Any, optional): The vectorizer to use. Defaults to None.
    Returns:
    Tuple[scipy.sparse.spmatrix, Any]: Tuple containing the n-gram matrix and the vectorizer used.
    """

    ngram_range = (1, 2)
    if vectorizer is None:
        # Initialize CountVectorizer with desired n-gram range
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    # Fit and transform the text data to extract n-gram features
    ngram_matrix = vectorizer.transform(texts)
    return ngram_matrix, vectorizer


def extract_dependency_features_for_corpus(texts, train_dependency_matrix=None) -> np.ndarray:
    """
    Extract dependency features from the given texts.
    Args:
    texts (list): List of texts to extract features from.
    train_dependency_matrix (np.ndarray, optional): Dependency matrix from the training data. Defaults to None.
    Returns:
    np.ndarray: Binary matrix containing dependency features.
    """

    try:
        nlp = spacy.load("en_core_web_sm")
    except ImportError as e:
        print("Error loading SpaCy model:", e)
        return None

    dependency_matrix = []
    all_features = set()

    for text in texts:
        doc = nlp(text)
        features = []
        for token in doc:
            # Extract features from the dependency tree
            features.append(token.dep_)
            all_features.add(token.dep_)  # Collect all unique features
        dependency_matrix.append(features)

    # Use the features from the training set if provided
    if train_dependency_matrix is not None:
        all_features = set(np.nonzero(train_dependency_matrix)[1])

    # Convert dependency features to a binary matrix
    binary_matrix = np.zeros((len(texts), len(all_features)), dtype=int)
    feature_index_map = {feature: i for i, feature in enumerate(sorted(all_features))}

    for i, doc_features in enumerate(dependency_matrix):
        for feature in doc_features:
            # Check if the feature exists in the index map
            if feature in feature_index_map:
                binary_matrix[i, feature_index_map[feature]] = 1
            else:
                # Handle the case when the feature is not found (e.g., 'ROOT')
                # Assign a default index or ignore it
                pass

    return binary_matrix


def extract_topic_features(texts, vectorizer, num_topics=5, num_words=5) -> Tuple[List[List[str]], np.ndarray]:
    """
    Extract topic features using Latent Dirichlet Allocation (LDA).
    Args:
    texts (list): List of texts to extract features from.
    vectorizer (Any): Vectorizer object to transform the text data.
    num_topics (int, optional): Number of topics. Defaults to 5.
    num_words (int, optional): Number of words per topic. Defaults to 5.
    Returns:
    Tuple[List[List[str]], np.ndarray]: Tuple containing top words for each topic and topic distributions.
    """

    # Transform the text data using the existing vectorizer
    X = vectorizer.transform(texts)

    # Initialize Latent Dirichlet Allocation (LDA) model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)

    # Fit LDA model to the data
    lda.fit(X)

    # Extract topic distributions for the text data
    topic_distributions = lda.transform(X)

    # Get the top words for each topic
    top_words = []
    feature_names = np.array(vectorizer.get_feature_names_out())
    for topic_idx, topic in enumerate(lda.components_):
        top_words.append([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]])

    return top_words, topic_distributions
