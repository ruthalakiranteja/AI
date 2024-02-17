import string
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def preprocess_data(df):
    # Remove punctuation
    df['review'] = df['review'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))

    # Convert to lowercase
    df['review'] = df['review'].apply(lambda x: x.lower())

    # Remove stopwords
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))

    # Tokenization and stemming
    stemmer = SnowballStemmer('english')
    df['review'] = df['review'].apply(lambda tokens: ' '.join([stemmer.stem(token) for token in word_tokenize(tokens)]))

    # Tokenization and lemmatization
    lemmatizer = WordNetLemmatizer()
    df['review'] = df['review'].apply(lambda tokens: ' '.join([lemmatizer.lemmatize(token) for token in word_tokenize(tokens)]))

    # Feature Engineering
    df['text_length'] = df['review'].apply(len)
    df['word_count'] = df['review'].apply(lambda x: len(x.split()))

def train_model(df):
    # Preprocess the data
    preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(df[['review', 'text_length', 'word_count']], df['sentiment'], test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['review'])
    X_val_tfidf = tfidf_vectorizer.transform(X_val['review'])

    # Apply Truncated SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=100)
    X_train_tfidf_reduced = svd.fit_transform(X_train_tfidf)
    X_val_tfidf_reduced = svd.transform(X_val_tfidf)

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_features_scaled = scaler.fit_transform(X_train[['text_length', 'word_count']])
    X_train_features = np.hstack([X_train_tfidf_reduced, X_train_features_scaled])

    X_val_features_scaled = scaler.transform(X_val[['text_length', 'word_count']])
    X_val_features = np.hstack([X_val_tfidf_reduced, X_val_features_scaled])

    # Random Forest with GridSearchCV
    param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_features, y_train)

    # Print best hyperparameters and accuracy
    print("Best hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    print("Best accuracy:", grid_search.best_score_)

    # Train final model with best hyperparameters on the full training set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_features, y_train)

    # Evaluate the model's performance on the validation set
    X_val_tfidf = tfidf_vectorizer.transform(X_val['review'])
    X_val_tfidf_reduced = svd.transform(X_val_tfidf)
    X_val_features_scaled = scaler.transform(X_val[['text_length', 'word_count']])
    X_val_features = np.hstack([X_val_tfidf_reduced, X_val_features_scaled])

    # Make predictions on the validation set
    y_val_pred = best_model.predict(X_val_features)

    # Evaluate the model's performance on the validation set
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    return tfidf_vectorizer, svd, scaler, best_model

def predict_sentiment(tfidf_vectorizer, svd, scaler, model):
    user_input = input("Enter a movie review: ")

    # Preprocess the user input
    user_df = pd.DataFrame({'review': [user_input], 'text_length': [len(user_input)], 'word_count': [len(user_input.split())]})
    preprocess_data(user_df)

    # TF-IDF Vectorization
    user_tfidf = tfidf_vectorizer.transform(user_df['review'])

    # Apply Truncated SVD for dimensionality reduction
    user_tfidf_reduced = svd.transform(user_tfidf)

    # Standardize features using StandardScaler
    user_features_scaled = scaler.transform(user_df[['text_length', 'word_count']])
    user_features = np.hstack([user_tfidf_reduced, user_features_scaled])

    # Make prediction
    prediction = model.predict(user_features)[0]
    
    print("Predicted sentiment:", prediction)

if __name__ == "__main__":
    # Load the model
    tfidf_vectorizer, svd, scaler, best_model = train_model(pd.read_csv('IMDB Dataset.csv'))

    # Use the model to predict sentiment based on user input
    predict_sentiment(tfidf_vectorizer, svd, scaler, best_model)
