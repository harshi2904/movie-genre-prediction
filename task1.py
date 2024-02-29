import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Function to read text files from a directory
def read_text_files(directory):
    texts = []  # Use a different variable name to store texts
    labels = []  # Use a different variable name to store labels
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):  # Assuming the text files have a .txt extension
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)  # Append each text to the list of texts
                # Extract label from file path or file name and append to labels list
                labels.append(os.path.basename(root))
    return texts, labels

# Step 1: Load the text data
data_dir = "Genre Classification Dataset"
X, y = read_text_files(data_dir)

# Step 2: Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for computational efficiency
X = tfidf_vectorizer.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = MultinomialNB()  # Using Naive Bayes classifier
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))



