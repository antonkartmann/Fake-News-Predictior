# Fake News Predictor

This project is a **Fake News Predictor** that uses Python, Natural Language Processing (NLP), and Machine Learning (Naive Bayes) to distinguish between real and fake news.

## Features

- **Data Processing & Cleaning:**  
  - Loads and combines "True.csv" and "Fake.csv" datasets.
  - Cleans text by converting to lowercase, removing numbers, punctuation, and stopwords.
  
- **Feature Extraction:**  
  - Converts text data into numerical features using TF-IDF vectorization.

- **Model Training:**  
  - Trains a Naive Bayes classifier.
  - Splits the data into training and test sets.
  - Evaluates the model using accuracy and a classification report.
  - Saves the trained model and TF-IDF vectorizer for later use.

- **Interactive Prediction:**  
  - Allows users to input any news article via the command line.
  - Outputs the prediction ("Fake News" or "Real News") along with prediction probabilities.

## Prerequisites

- **Python 3.12** (or a compatible version)
- The following Python libraries:
  - `pandas`
  - `nltk`
  - `scikit-learn`
  - `joblib`

## Installation & Setup

1. **Clone the Repository (optional):**
   ```bash
   git clone https://github.com/antonkartmann/Fake-News-Predictior
   cd Fake-News-Predictor
