# Email/SMS Spam Classifier

This project is a simple web application for classifying email or SMS messages as spam or not spam. It is built using Streamlit for the web interface and uses a pre-trained machine learning model for the classification.

![Spam Classifier Screenshot]([Screenshot_2024-06-30_141219.png](https://github.com/aadityakolhapure/SpamNotSpam/blob/main/Screenshot%202024-06-30%20141219.png))


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Features

- User-friendly web interface using Streamlit.
- Preprocessing of input text including tokenization, stopword removal, and stemming.
- Classification using a pre-trained TF-IDF vectorizer and machine learning model.
- Real-time prediction of spam or not spam.

## Installation

### Prerequisites

- Python 3.x
- Streamlit
- NLTK
- scikit-learn
- pickle

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/spam-classifier.git
    cd spam-classifier
    ```

2. Install the required Python packages:
    ```bash
    pip install streamlit nltk scikit-learn
    ```

3. Download necessary NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

4. Place the pre-trained `vectorizer.pkl` and `model.pkl` files in the project directory.

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter the email or SMS text in the provided text area and click on the "Predict" button to see the classification result.

## Model Training

If you want to train your own model, follow these steps:

1. Prepare a dataset of labeled email/SMS messages.
2. Preprocess the text data using the same steps as in the `transform_text` function.
3. Train a machine learning model (e.g., Logistic Regression, SVM) using a TF-IDF vectorizer.
4. Save the trained model and vectorizer using `pickle`.

