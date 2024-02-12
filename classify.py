import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

# nltk.download('punkt')
# nltk.download('stopwords')



def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(tokens)


def select_instances_per_label(corpus_file, output_file, instances_per_label=2):
    selected_instances = {}
    label_counts = {}

    # Read corpus file into DataFrame
    try:
        corpus_df = pd.read_csv(corpus_file, sep='\t', header=None, names=['File', 'Label', 'Text'])
    except Exception as e:
        print(f"Error reading corpus file: {e}")
        return

    # Iterate through corpus DataFrame
    for _, row in corpus_df.iterrows():
        file_name, label, text = row['File'], row['Label'], row['Text']
        if label not in selected_instances:
            selected_instances[label] = []
            label_counts[label] = 0

        if label_counts[label] < instances_per_label:
            selected_instances[label].append((file_name, text))
            label_counts[label] += 1

    # Write selected instances to output file
    with open(output_file, 'w') as f_out:
        for label, instances in selected_instances.items():
            for file_name, text in instances:
                f_out.write(f"{file_name}\t{label}\t{text}\n")

stop_words = set(stopwords.words('english'))

# Read annotated corpus from file
corpus_df = pd.read_csv("all_annotated_texts.txt", sep='\t', header=None, names=['File', 'Label', 'Text'])
# Make sample corpus for development purposes
sample_corpus = select_instances_per_label('all_annotated_texts.txt', 'sample_annotated_corpus.txt')
corpus_df = pd.read_csv("sample_annotated_corpus.txt", sep='\t', header=None, names=['File', 'Label', 'Text'])

# Encode labels
mlb = MultiLabelBinarizer()
labels = [set(x.split(',')) for x in corpus_df['Label']]
# print(labels)
encoded_labels = mlb.fit_transform(labels)
# print(encoded_labels)

# Split the data into training, development, and test sets
X_train_dev, X_test, y_train_dev, y_test = train_test_split(corpus_df['Text'], encoded_labels, test_size=0.2,
                                                            random_state=42, stratify=encoded_labels)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.25, random_state=42,
                                                  stratify=y_train_dev)


# Preprocess the text: tokenize and remove stop words
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

# TRANSFORMERS
# Load pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize text data
X_train_tokenized = tokenizer(X_train_processed.tolist(), padding=True, truncation=True, return_tensors="pt")
X_dev_tokenized = tokenizer(X_dev_processed.tolist(), padding=True, truncation=True, return_tensors="pt")

# Convert labels to PyTorch tensors
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_dev_tensor = torch.tensor(y_dev, dtype=torch.float32)

# Define DataLoader for training and development sets
train_dataset = torch.utils.data.TensorDataset(X_train_tokenized.input_ids, X_train_tokenized.attention_mask,
                                               y_train_tensor)
dev_dataset = torch.utils.data.TensorDataset(X_dev_tokenized.input_ids, X_dev_tokenized.attention_mask, y_dev_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

# Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(mlb.classes_))

# Fine-tune the model
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Avg. Train Loss: {avg_train_loss}")

# Evaluate on development set
model.eval()
all_dev_preds = []
all_dev_labels = []
with torch.no_grad():
    for batch in dev_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.sigmoid(logits)
        all_dev_preds.append(preds.cpu().detach().numpy())
        all_dev_labels.append(labels.cpu().detach().numpy())

y_pred_dev_transformers = torch.cat(all_dev_preds, dim=0)
y_pred_dev_transformers[y_pred_dev_transformers >= 0.5] = 1
y_pred_dev_transformers[y_pred_dev_transformers < 0.5] = 0
y_pred_dev_transformers = y_pred_dev_transformers.numpy()
print("Classification Report (Transformers):\n", classification_report(y_dev, y_pred_dev_transformers))

