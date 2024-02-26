import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def check_claim_verbs(sentence: list[str]) -> int:
    claim_verbs = ['argue', 'claim', 'emphasise', 'contend', 'maintain', 'assert', 'theorize', 'support the view that',
                   'deny', 'negate', 'refute', 'reject', 'challenge', 'strongly believe that', 'counter the view that',
                   'acknowledge', 'consider', 'discover', 'hypothesize', 'object', 'say', 'admit', 'assume', 'decide',
                   'doubt', 'imply', 'observe', 'show', 'agree', 'believe', 'demonstrate', 'emphasize', 'indicate',
                   'point out', 'state', 'allege', 'argument that']
    for lemma in sentence:
        if lemma in claim_verbs:
            return 1
    return 0


def word2features(sent, i):
    """Return list of features for every token in a sentence."""
    features = []
    word = sent[i]

    features.append(word.lower())
    features = [f.encode('utf-8') for f in features]
    features.append(word[-3:])
    features.append(word[-2:])
    features.append(str(word.istitle()))
    features.append(str(word.isupper()))
    features.append(str(word.istitle()))
    features.append(str(word.isdigit()))

    if i > 0:
        features.append(sent[i - 1])
    else:
        features.append(str('BOS'))
    punc_cat = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
    # Comment out features that give us worse accuracy
    # if all(unicodedata.category(x) in punc_cat for x in word):
    #     features.append("PUNCTUATION")
    # if i < len(sent) - 1:
    #     features.append(sent[i + 1])
    # else:
    #     features.append(str('EOS'))

    return features


def sent2features(sent, index):
    """Call word2features on sentence."""
    return word2features(sent, index)


# Function to extract n-gram features for multiple texts
def extract_ngram_features_for_corpus(texts, vectorizer=None):
    ngram_range = (1, 2)
    if vectorizer is None:
        # Initialize CountVectorizer with desired n-gram range
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    # Fit and transform the text data to extract n-gram features
    ngram_matrix = vectorizer.transform(texts)
    return ngram_matrix, vectorizer


# Function to extract dependency features for multiple texts
def extract_dependency_features_for_corpus(texts):
    # Implement dependency parsing and extract features here
    # Placeholder implementation
    dependency_matrix = np.zeros((len(texts), 10))  # Example: 10 features per text
    return dependency_matrix
