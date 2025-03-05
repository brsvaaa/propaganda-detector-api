# Импорт библиотек

import os
'''
# Disable GPU usage by setting CUDA_VISIBLE_DEVICES to an empty string
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''
# GCS Bucket Configuration
BUCKET_NAME = "propdetector_models"  # Replace with your actual GCS bucket name
MODEL_DIR = "models"  # Local directory to store downloaded models
# Ensure the local model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
# Set TensorFlow logging level to 2 to reduce verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import tf_keras as keras
import numpy as np
from transformers import XLNetTokenizer, TFAutoModelForSequenceClassification, RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from flask import Flask, request, jsonify
from google.cloud import storage
from flask_cors import CORS
import threading
import logging

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}) # This enables CORS for all routes by default

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# The heavy initialization code is commented out for now.
# This will allow the /health endpoint to respond quickly.
def download_folder_from_gcs(bucket, prefix, local_dir):
    """
    Downloads all blobs in GCS with the given prefix (folder) to a local directory.
    Skips blobs that represent directories.
    """
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        # Skip blobs that are "directories" (end with a slash)
        if blob.name.endswith("/"):
            continue

        # Remove the prefix from the blob name to create a relative path.
        relative_path = blob.name[len(prefix):].lstrip("/")
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(local_path)
    print(f"✅ Folder '{prefix}' downloaded to '{local_dir}'.")

def init_models():
    # Google Cloud Storage Client
    storage_client = storage.Client(project='iconic-market-452120-a0')
    
    # List of models to download
    MODEL_FILES = [
        "Appeal_to_Authority_model.keras",
        "Bandwagon_Reductio_ad_hitlerum_model.keras",
        "Black-and-White_Fallacy_model.keras",
        "Causal_Oversimplification_model.keras",
        "Slogans_model.keras",
        "Thought-terminating_Cliches_model.keras",
        "text_classification_model.keras",
        "vectorizer.joblib",
        "label_encoder.joblib"
    ]
    
    def download_from_gcs(blob_name, destination_file):
        """Downloads a file from GCS to the local directory if it doesn't exist."""
        local_path = os.path.join(MODEL_DIR, destination_file)
        if not os.path.exists(local_path):
            print(f"Downloading {blob_name} from GCS...")
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            print(f"✅ Downloaded {blob_name} to {local_path}")
        else:
            print(f"✔ {blob_name} already exists locally.")
    
    # Download all required models
    for model_file in MODEL_FILES:
        download_from_gcs(model_file, model_file)
    
    # Download model folders for Transformers models
    bucket = storage_client.bucket(BUCKET_NAME)
    # Define the folder names as stored in GCS (ensure these match exactly)
    xlnet_prefix = "xlnet_trained_model/"   # Include trailing slash
    roberta_prefix = "roberta_trained_model/" # Include trailing slash

    # Local destination directories for the models
    xlnet_local = os.path.join(MODEL_DIR, "xlnet_trained_model")
    roberta_local = os.path.join(MODEL_DIR, "roberta_trained_model")
    
    os.makedirs(xlnet_local, exist_ok=True)
    os.makedirs(roberta_local, exist_ok=True)
    
    download_folder_from_gcs(bucket, xlnet_prefix, xlnet_local)
    download_folder_from_gcs(bucket, roberta_prefix, roberta_local)
    
    print("🔄 Loading models...")
    global tfidf_vectorizer, label_encoder
    global keras_model, authority_model, bandwagon_model, black_model, causal_model, slogans_model, thought_model
    global xlnet_tokenizer, xlnet_model, roberta_tokenizer, roberta_model

    # Load Vectorizer & Label Encoder
    tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # Load Keras models using tf_keras
    keras_model = keras.models.load_model(os.path.join(MODEL_DIR, "text_classification_model.keras"))
    authority_model = keras.models.load_model(os.path.join(MODEL_DIR, "Appeal_to_Authority_model.keras"))
    bandwagon_model = keras.models.load_model(os.path.join(MODEL_DIR, "Bandwagon_Reductio_ad_hitlerum_model.keras"))
    black_model = keras.models.load_model(os.path.join(MODEL_DIR, "Black-and-White_Fallacy_model.keras"))
    causal_model = keras.models.load_model(os.path.join(MODEL_DIR, "Causal_Oversimplification_model.keras"))
    slogans_model = keras.models.load_model(os.path.join(MODEL_DIR, "Slogans_model.keras"))
    thought_model = keras.models.load_model(os.path.join(MODEL_DIR, "Thought-terminating_Cliches_model.keras"))

    # Load Transformers models
    xlnet_model_path = os.path.join(MODEL_DIR, "xlnet_trained_model")
    roberta_model_path = os.path.join(MODEL_DIR, "roberta_trained_model")
    xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    xlnet_model = TFAutoModelForSequenceClassification.from_pretrained(xlnet_model_path)
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_model = TFRobertaForSequenceClassification.from_pretrained(roberta_model_path)
    
    print("✅ Models loaded.")
'''
# Start heavy initialization in a background thread so that the server starts quickly
threading.Thread(target=init_models, daemon=True).start()
'''

# =========================
# Настройка spaCy для разбиения текста
# =========================
nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    """Разбивает текст на предложения с помощью spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# =========================
# Функции для предсказаний
# =========================

def predict_xlnet(text):
    """Предсказание для XLNet."""
    inputs = xlnet_tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    logits = xlnet_model(**inputs).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()
    return probs

def predict_roberta(text):
    """Предсказание для RoBERTa."""
    inputs = roberta_tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    logits = roberta_model(**inputs).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()
    return probs

def pad_vectorized_text(vectorized_text, expected_shape):
    """Дополняет или обрезает вектор до нужной формы."""
    current_shape = vectorized_text.shape[1]
    if current_shape < expected_shape:
        padding = np.zeros((vectorized_text.shape[0], expected_shape - current_shape))
        return np.hstack((vectorized_text, padding))
    return vectorized_text[:, :expected_shape]

def predict_keras(text):
    """Предсказание для модели Keras."""
    vectorized_text = tfidf_vectorizer.transform([text]).toarray()
    vectorized_text = pad_vectorized_text(vectorized_text, 15396)
    probs = keras_model.predict(vectorized_text)
    return probs

def softmax(x):
    """Функция softmax для логитов."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def ensemble_multiclass_predict(text):
    """Ансамблевое предсказание для мультиклассовых моделей."""
    p_xlnet = predict_xlnet(text)
    p_roberta = predict_roberta(text)
    p_keras = predict_keras(text)

    # Применяем softmax
    p_xlnet = softmax(p_xlnet)
    p_roberta = softmax(p_roberta)
    p_keras = softmax(p_keras)

    avg_probs = (p_xlnet + p_roberta + p_keras) / 3
    final_prediction = np.argmax(avg_probs, axis=1)
    return final_prediction[0], avg_probs

# =========================
# Запуск Flask API
# =========================
from flask_cors import cross_origin
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response
@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    return jsonify({"message": "Endpoint working."})
    # Uncomment the block below for actual prediction logic.
    '''
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    sentences = split_sentences(text)
    results = []
    for sentence in sentences:
        multiclass_prediction, multiclass_probs = ensemble_multiclass_predict(sentence)
        results.append({
            "sentence": sentence,
            "Multiclass Prediction": multiclass_prediction,
        })
    return jsonify({"results": results})
    '''

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("Starting server on port:", port)
    app.run(host='0.0.0.0', port=port)



