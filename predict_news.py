import joblib
import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text and remove paragraph breaks
def clean_text(text):
    text = text.lower()
    # Replace one or more whitespace characters (including newlines) with a single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Load the saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to make a prediction on a news article
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    probabilities = model.predict_proba(vect_text)[0]
    label = "Real News" if prediction == 1 else "Fake News"
    return label, probabilities

# Interactive prediction loop
if __name__ == '__main__':
    print("Fake News Predictor - Enter a news article (or type 'exit' to quit):")
    while True:
        user_input = input(">>> ")
        if user_input.lower() == 'exit':
            break
        label, probs = predict_news(user_input)
        print(f"Prediction: {label}")
        print(f"Probabilities -> Fake News: {probs[0]:.2f}, Real News: {probs[1]:.2f}\n")

