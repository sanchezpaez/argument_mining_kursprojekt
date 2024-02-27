import numpy as np
import click.termui
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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


def extract_ngram_features_for_corpus(texts, vectorizer=None):
    ngram_range = (1, 2)
    if vectorizer is None:
        # Initialize CountVectorizer with desired n-gram range
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    # Fit and transform the text data to extract n-gram features
    ngram_matrix = vectorizer.transform(texts)
    return ngram_matrix, vectorizer


def extract_dependency_features_for_corpus(texts):
    try:
        nlp = spacy.load("en_core_web_sm")
    except ImportError as e:
        print("Error loading SpaCy model:", e)
        return None

    dependency_matrix = []
    for text in texts:
        doc = nlp(text)
        features = []
        for token in doc:
            # Extract features from the dependency tree
            features.append(token.dep_)
        dependency_matrix.append(features)
    return dependency_matrix


def extract_topic_features(texts, vectorizer, num_topics=5, num_words=5):
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
