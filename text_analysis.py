import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# ========== 1. LOAD DATASET ==========
# You can use any text dataset (SMS Spam, Movie Reviews, News, etc.)
# Example: SMS Spam Dataset (download from UCI or Kaggle)
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nLabel Distribution:\n", df['label'].value_counts())

# ========== 2. TEXT PREPROCESSING ==========

# Convert to lowercase
df['cleaned'] = df['text'].str.lower()

# Remove special characters and numbers
df['cleaned'] = df['cleaned'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# Remove extra whitespace
df['cleaned'] = df['cleaned'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

print("\nAfter Cleaning:\n", df[['text', 'cleaned']].head())

# ========== 3. FEATURE EXTRACTION (TF-IDF) ==========

tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X = tfidf.fit_transform(df['cleaned'])
y = df['label']

print("\nTF-IDF Matrix Shape:", X.shape)

# ========== 4. TRAIN-TEST SPLIT ==========

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 5. MODEL - NAIVE BAYES ==========

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== 6. EVALUATION ==========

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ========== 7. VISUALIZATION ==========

# Label distribution
df['label'].value_counts().plot(kind='bar', color=['green', 'red'], edgecolor='black')
plt.title("Spam vs Ham Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Word count distribution
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 5))
df[df['label'] == 'ham']['word_count'].hist(bins=30, alpha=0.7, label='Ham', color='green')
df[df['label'] == 'spam']['word_count'].hist(bins=30, alpha=0.7, label='Spam', color='red')
plt.title("Word Count Distribution: Spam vs Ham")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# ========== 8. TEST WITH CUSTOM INPUT ==========

sample_texts = ["Congratulations! You won a free ticket!", "Hey, are we meeting tomorrow?"]
sample_tfidf = tfidf.transform(sample_texts)
predictions = model.predict(sample_tfidf)

for text, pred in zip(sample_texts, predictions):
    print(f"Text: '{text}' => Prediction: {pred}")

# ========== INFERENCE ==========
"""
INFERENCE:
1. The dataset contains SMS messages labeled as 'spam' or 'ham' (not spam).
2. Text preprocessing (lowercasing, removing special chars) cleans the raw text.
3. TF-IDF converts text into numerical features based on word importance.
4. Multinomial Naive Bayes is ideal for text classification tasks.
5. The model achieves high accuracy in distinguishing spam from ham messages.
6. Spam messages tend to have more words and contain promotional keywords.
7. The confusion matrix shows very few false positives/negatives.
"""
