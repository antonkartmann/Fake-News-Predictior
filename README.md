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
   git clone https://github.com/antonkartmann/Fake-News-Predictor
   cd Fake-News-Predictor


## Example Usage

**Example with a BBC Article** https://www.bbc.com/news/articles/cgkm0k2j6edo

**Input:**
Fake News Predictor - Enter a news article (or type 'exit' to quit):
>>> A group of scientists due to work together for months at a remote Antarctic research station has been rocked after a member of the team was accused of physical assault. A team of nine researchers were due to spend the Antarctic winter at the South African-run base, which sits about 170km (about 105 miles) from the edge of the ice shelf and is difficult to reach. But a spokesperson for the South African government told the BBC "there was an assault" at the station, following earlier allegations of inappropriate behaviour from inside the camp.In a further message seen by the BBC, the South African environment ministry said it was responding to the concerns with "utmost urgency". South Africa's Sunday Times, which was first to report the story, said members of the team had pleaded to be rescued. The ministry said that those in the team had been subject to "a number of evaluations that include background checks, reference checks, medical assessment as well as a psychometric evaluation by qualified professionals", which all members had cleared. In a subsequent statement, the ministry added that it was "not uncommon" for individuals to have an initial adjustment when they arrive at extremely remote areas even if assessments showed no areas of concern. It said when the vessel departed for Antarctica on 1 February "all was in order", and the incident was first reported to the ministry on 27 February. The statement added the department "immediately activated the response plan in order to mediate and restore relations at the base". "This process has been ongoing on an almost daily basis in order to ensure that those on the base know that the Department is supportive and willing to do whatever is needed to restore the interpersonal relationships, but also firm in dealing with issues of discipline," it said. The department said allegations of sexual harassment were also being investigated, but that reports of sexual assault were incorrect. The department added that a government minister was personally handling the incident, and the alleged perpetrator had "willingly participated in further psychological evaluation, has shown remorse and is willingly cooperative to follow any interventions that are recommended". The alleged perpetrator has also written a formal apology to the victim, it said. The Sanae IV research base is located more than 4,000km from mainland South Africa and harsh weather conditions mean scientists can be cut off there for much of the year. The base typically houses staff who stay through the Antarctic winter for approximately 13 months.<<<


**Output:**
Prediction: Real News
Probabilities -> Fake News: 0.09, Real News: 0.91
