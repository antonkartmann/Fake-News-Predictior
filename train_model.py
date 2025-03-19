import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    words = text.split()
    filtered_words = []  # Create an empty list to store filtered words
    for word in words:  # Iterate through each word in the original list
        if word not in stop_words:  # Check if the word is NOT in the stopwords set
            filtered_words.append(word)  # Add it to the new list
    words = filtered_words  # Overwrite the original list with the filtered words

    return ' '.join(words)

# 1. Load data
# Make sure that "True.csv" and "Fake.csv" are in the same directory as this script.
df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

# Assign labels: Real News = 1, Fake News = 0
df_true['label'] = 1
df_fake['label'] = 0

# Combine the datasets
df = pd.concat([df_true, df_fake]).reset_index(drop=True)
# I use both the title and the text; you can also choose to use only the text.
df = df[['title', 'text', 'label']]
df['combined'] = df['title'] + " " + df['text']

# 2. Data cleaning: Clean the text
df['clean_text'] = df['combined'].apply(clean_text)

# 3. Convert text to numerical values using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# 4. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the model and vectorizer for later use
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer have been saved.")
