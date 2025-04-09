from flask import Flask, jsonify, request
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import re
import os

# Flask app
app = Flask(__name__)
CORS(app)

# Load model
model_predict = joblib.load('best_urgency_classifier.pkl')

# NLTK downloads
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Department keywords
department_keywords = {
    'Infrastructure': ['infrastructure', 'bridge', 'road', 'water supply', 'sewer', 'drainage', 'electricity', 'power', 'internet', 'broadband', 'maintenance', 'pothole', 'construction', 'utility', 'network'],
    'Public Safety': ['police', 'crime', 'safety', 'security', 'emergency', 'fire', 'ambulance', 'rescue', 'patrol', 'violence', 'protection', 'law enforcement'],
    'Healthcare': ['hospital', 'doctor', 'nurse', 'medical', 'patient', 'health', 'care', 'disease', 'treatment', 'vaccine', 'medicine', 'clinic'],
    'Environment': ['environment', 'pollution', 'climate', 'green', 'forest', 'wildlife', 'conservation', 'nature', 'park', 'waste', 'recycling', 'sustainability', 'emissions'],
    'Education': ['school', 'student', 'teacher', 'education', 'curriculum', 'college', 'university', 'classroom', 'learn', 'campus', 'textbook', 'tuition'],
    'Transportation': ['road', 'highway', 'traffic', 'parking', 'bus', 'transit', 'transportation', 'pedestrian', 'sidewalk', 'bicycle', 'vehicle', 'metro', 'rail', 'public transport'],
    'Housing': ['housing', 'apartment', 'rent', 'mortgage', 'homeless', 'tenant', 'landlord', 'property', 'affordable', 'building', 'residence', 'shelter'],
    'Economic Development': ['finance', 'tax', 'budget', 'funding', 'revenue', 'expense', 'economic', 'money', 'grant', 'fiscal', 'financial', 'investment', 'jobs', 'employment', 'industry', 'startup', 'development'],
    'Social Services': ['welfare', 'social', 'services', 'childcare', 'disability', 'elderly', 'pension', 'benefits', 'support', 'family', 'assistance', 'food', 'shelter', 'counseling']
}

# Text preprocessing
def preprocess_text(text, return_tokens: bool = False):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens if return_tokens else ' '.join(tokens)

# Department classification
def classify_petition(petition_text: str):
    tokens = preprocess_text(petition_text, return_tokens=True)
    processed_text = ' '.join(tokens)
    lemmatizer = WordNetLemmatizer()

    department_scores = {}
    for dept, keywords in department_keywords.items():
        score = 0
        for keyword in keywords:
            # Lemmatize the keyword too
            keyword_tokens = nltk.word_tokenize(keyword.lower())
            keyword_tokens = [lemmatizer.lemmatize(kw) for kw in keyword_tokens]
            lemmatized_keyword = ' '.join(keyword_tokens)
            if lemmatized_keyword in processed_text:
                score += 1
        department_scores[dept] = score

    best_department = max(department_scores, key=department_scores.get)
    if department_scores[best_department] == 0:
        best_department = "Other"
    return best_department

# Processing function
def process_petition(petition_text: str):
    dept = classify_petition(petition_text)
    return {"department": dept}

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        description = data.get('description', '')

        if not description:
            return jsonify({'error': 'No description provided'}), 400

        clean_desc = preprocess_text(description)
        prediction = model_predict.predict([clean_desc])[0]
        confidence = max(model_predict.predict_proba([clean_desc])[0])

        result = process_petition(description)

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'department': result['department']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True, port=5001)
    port = int(os.environ.get('PORT', 5000))  # Render sets PORT automatically
    app.run(debug=False, host='0.0.0.0', port=port)  # 0.0.0.0 allows external access
