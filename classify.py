import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Read annotated corpus from file
corpus_df = pd.read_csv("all_annotated_texts.txt", sep='\t', header=None, names=['File', 'Label', 'Text'])

# Encode labels
mlb = MultiLabelBinarizer()
labels = [set(x.split(',')) for x in corpus_df['Label']]
# print(labels)
encoded_labels = mlb.fit_transform(labels)
# print(encoded_labels)

# Split the data into training, development, and test sets
X_train_dev, X_test, y_train_dev, y_test = train_test_split(corpus_df['Text'], encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.25, random_state=42, stratify=y_train_dev)

# Preprocess the text: tokenize the text and remove stop words


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(tokens)


X_train_processed = X_train.apply(preprocess_text)
X_dev_processed = X_dev.apply(preprocess_text)
X_test_processed = X_test.apply(preprocess_text)

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train_processed)
X_dev_vectorized = vectorizer.transform(X_dev_processed)
X_test_vectorized = vectorizer.transform(X_test_processed)

# Save processed data into files
# Save vectorized features into separate files
pd.DataFrame(X_train_vectorized.toarray()).to_csv('X_train_vectorized.csv', index=False)
pd.DataFrame(X_dev_vectorized.toarray()).to_csv('X_dev_vectorized.csv', index=False)
pd.DataFrame(X_test_vectorized.toarray()).to_csv('X_test_vectorized.csv', index=False)

# Save labels into separate files
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_dev).to_csv('y_dev.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

print(X_train_processed)

# Train scikit-learn classifier
clf_sklearn = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf_sklearn.fit(X_train_vectorized, y_train)

# Evaluate on development set
y_pred_dev_sklearn = clf_sklearn.predict(X_dev_vectorized)
print("Classification Report (scikit-learn):\n", classification_report(y_dev, y_pred_dev_sklearn, zero_division=1))
