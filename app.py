from flask import Flask, render_template, request
import pandas as pd
from sentiment_analysis import preprocess_data, train_model, predict_sentiment

app = Flask(__name__)

# Load the model during app startup
tfidf_vectorizer, svd, scaler, best_model = train_model(pd.read_csv('IMDB Dataset.csv'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Preprocess user input and make prediction
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
        prediction = best_model.predict(user_features)[0]

        return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

