from flask import Flask, jsonify, request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import re
import google.generativeai as genai
import os
from typing import Dict, List, Tuple
import google.api_core.exceptions
import google.generativeai.types.generation_types
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# app = Flask(__name__)

model_predict = joblib.load('best_urgency_classifier.pkl')

genai.configure(api_key="AIzaSyCUNNh0Iy9uJ6a8Dy9ht3wmg7OO-SrQT8Q")
model = genai.GenerativeModel('gemini-1.5-pro')
departments = [
    'Infrastructure',
    'Public Safety',
    'Healthcare',
    'Environment',
    'Education',
    'Transportation',
    'Housing',
    'Economic Development',
    'Social Services',
    'Other'
  ]

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def classify_petition(petition_text: str) -> str:
    """Classify the petition into the most relevant department."""
    if not petition_text:
        return "Other"

    prompt = f"""
    Classify the following petition into EXACTLY ONE of these departments:
    if the petion is based or related to highways, roads, street or anything related to transport comes under the department "Transportation"
    "Infrastructure" only applicable to the damage of the public buildings such as damage in government office,etc.,.
    {', '.join(departments)}

    Petition:
    {petition_text}

    Return ONLY the department name, nothing else.
    """

    try:
        response = model.generate_content(prompt)
        department = response.text.strip().lower()

        for dept in departments:
            if dept.lower() in department:
                return dept
        return "Other"

    # except (google.api_core.exceptions.GoogleAPIError, google.generativeai.types.generation_types.GenerationError) as e:
    #     print(f"Error in classification: {e}")
    #     return "Other"
    except Exception as e:
        print(f"Unexpected error in classification: {e}")
        return "Other"

def summarize_petition(petition_text: str, max_words: int = 20) -> str:
    """Generate a concise summary of the petition."""
    if not petition_text:
        return "Summary unavailable"

    prompt = f"""
    Summarize the following petition in approximately {max_words} words.
    Ensure the summary is concise, complete, and self-contained. It must fully express the main idea of the petition without cutting off mid-sentence or leaving critical points incomplete.
    Do not exceed the word limit, but rephrase or condense ideas instead of omitting important information.


    Petition:
    {petition_text}

    Return ONLY the summary, nothing else.
    """

    try:
        response = model.generate_content(prompt)
        summary = response.text.strip()

        # words = summary.split()
        # if len(words) > max_words:
        #     summary = ' '.join(words[:max_words])

        return summary

    except Exception as e:
        print(f"Unexpected error in summarization: {e}")
        words = petition_text.split()
        return ' '.join(words[:min(max_words, len(words))])

def process_petition(petition_text: str) -> Dict:
    """Process a petition by classifying it and generating a summary."""
    department = classify_petition(petition_text)
    summary = summarize_petition(petition_text)

    return {
        "department": department,
        "summary": summary
    }

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    clean_text = ' '.join(tokens)
    return clean_text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        description = data.get('description','')

        if not description:
            return jsonify({'error': 'No description provided'}), 400

        clean_desc = preprocess_text(description)

        prediction = model_predict.predict([clean_desc])[0]
        confidence = max(model_predict.predict_proba([clean_desc])[0])
        result = process_petition(description)
        print(result)

        return jsonify({'prediction': prediction, 'confidence': confidence,'department':result['department'],'summary':result['summary']})
    
    # except google.api_core.exceptions.GoogleAPIError as e:
    #     print(f"Google API Error: {e}")
    #     return jsonify({'error': str(e)}), 500
    # except google.generativeai.GenerationError as e:
    #     print(f"Gemini Generation Error: {e}")
    #     return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}),500

if __name__ == '__main__':
    app.run(debug=True)
