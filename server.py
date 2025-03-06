'''
# Импорт библиотек
import os
# Optionally disable GPU usage:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# GCS Bucket Configuration
BUCKET_NAME = "propdetector_models"  # Replace with your actual GCS bucket name
MODEL_DIR = "models"  # Local directory to store downloaded models
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import tf_keras as keras
import numpy as np
from transformers import XLNetTokenizer, TFAutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from flask import Flask, request, jsonify
from google.cloud import storage
from flask_cors import CORS, cross_origin
import threading
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# Global flag to indicate model initialization status
models_loaded = False

def download_folder_from_gcs(bucket, prefix, local_dir):
    """
    Downloads all blobs in GCS with the given prefix (folder) to a local directory.
    Skips blobs that represent directories.
    """
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_path = blob.name[len(prefix):].lstrip("/")
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logging.info(f"Downloading {blob.name} to {local_path}...")
        try:
            blob.download_to_filename(local_path)
        except Exception as e:
            logging.error(f"Error downloading {blob.name}: {e}")
    logging.info(f"✅ Folder '{prefix}' downloaded to '{local_dir}'.")

def init_models():
    global models_loaded
    try:
        storage_client = storage.Client(project='iconic-market-452120-a0')
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
        def download_file(blob_name, destination_file):
            local_path = os.path.join(MODEL_DIR, destination_file)
            if not os.path.exists(local_path):
                try:
                    logging.info(f"Downloading {blob_name} from GCS...")
                    bucket = storage_client.bucket(BUCKET_NAME)
                    blob = bucket.blob(blob_name)
                    blob.download_to_filename(local_path)
                    logging.info(f"✅ Downloaded {blob_name} to {local_path}")
                except Exception as e:
                    logging.error(f"Error downloading {blob_name}: {e}")
            else:
                logging.info(f"✔ {blob_name} already exists locally.")
        for model_file in MODEL_FILES:
            download_file(model_file, model_file)
        bucket = storage_client.bucket(BUCKET_NAME)
        # Download folder for XLNet model only
        xlnet_prefix = "xlnet_trained_model/"   # Include trailing slash
        xlnet_local = os.path.join(MODEL_DIR, "xlnet_trained_model")
        os.makedirs(xlnet_local, exist_ok=True)
        try:
            download_folder_from_gcs(bucket, xlnet_prefix, xlnet_local)
        except Exception as e:
            logging.error(f"Error downloading folder {xlnet_prefix}: {e}")
        logging.info("🔄 Loading models...")
        global tfidf_vectorizer, label_encoder
        global keras_model, authority_model, bandwagon_model, black_model, causal_model, slogans_model, thought_model
        global xlnet_tokenizer, xlnet_model
        try:
            tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
        except Exception as e:
            logging.error("Error loading vectorizer: " + str(e))
        try:
            label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
        except Exception as e:
            logging.error("Error loading label_encoder: " + str(e))
        try:
            keras_model = keras.models.load_model(os.path.join(MODEL_DIR, "text_classification_model.keras"))
            authority_model = keras.models.load_model(os.path.join(MODEL_DIR, "Appeal_to_Authority_model.keras"))
            bandwagon_model = keras.models.load_model(os.path.join(MODEL_DIR, "Bandwagon_Reductio_ad_hitlerum_model.keras"))
            black_model = keras.models.load_model(os.path.join(MODEL_DIR, "Black-and-White_Fallacy_model.keras"))
            causal_model = keras.models.load_model(os.path.join(MODEL_DIR, "Causal_Oversimplification_model.keras"))
            slogans_model = keras.models.load_model(os.path.join(MODEL_DIR, "Slogans_model.keras"))
            thought_model = keras.models.load_model(os.path.join(MODEL_DIR, "Thought-terminating_Cliches_model.keras"))
        except Exception as e:
            logging.error("Error loading Keras models: " + str(e))
        try:
            xlnet_model_path = os.path.join(MODEL_DIR, "xlnet_trained_model")
            xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
            xlnet_model = TFAutoModelForSequenceClassification.from_pretrained(xlnet_model_path)
        except Exception as e:
            logging.error("Error loading XLNet models: " + str(e))
        logging.info("✅ Models loaded.")
        # Create a flag file indicating that models are loaded.
        flag_path = os.path.join(MODEL_DIR, "models_loaded.flag")
        with open(flag_path, "w") as f:
            f.write("true")
        models_loaded = True
    except Exception as e:
        logging.error("General error in init_models: " + str(e))

# Start heavy initialization in a background thread
threading.Thread(target=init_models, daemon=True).start()

# Настройка spaCy для разбиения текста
nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    """Разбивает текст на предложения с помощью spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# Prediction functions
def predict_xlnet(text):
    """Предсказание для XLNet."""
    inputs = xlnet_tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    logits = xlnet_model(**inputs).logits
    probs = np.array(tf.nn.softmax(logits, axis=1))
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
    """Ансамблевое предсказание для мультиклассовых моделей using XLNet and Keras only."""
    p_xlnet = predict_xlnet(text)
    p_keras = predict_keras(text)
    p_xlnet = softmax(p_xlnet)
    p_keras = softmax(p_keras)
    avg_probs = (p_xlnet + p_keras) / 2  # Average predictions from XLNet and Keras
    final_prediction = np.argmax(avg_probs, axis=1)
    return final_prediction[0], avg_probs

# Запуск Flask API
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
    
    logging.info(f"models_loaded flag: {models_loaded}")  # Debugging log
    
    # Check if the flag file exists.
    flag_path = os.path.join(MODEL_DIR, "models_loaded.flag")
    if not os.path.exists(flag_path):
        return jsonify({"error": "Models are still loading, please try again later."}), 503
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

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

    except Exception as e:
        logging.error("Error in /predict: " + str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info("Starting server on port: " + str(port))
    app.run(host='0.0.0.0', port=port)
'''
# Импорт библиотек
import os
# Optionally disable GPU usage:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# GCS Bucket Configuration
BUCKET_NAME = "propdetector_models"  # Replace with your actual GCS bucket name
MODEL_DIR = "models"                # Local directory to store downloaded models
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import tf_keras as keras
import numpy as np
from transformers import XLNetTokenizer, TFAutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from flask import Flask, request, jsonify
from google.cloud import storage
from flask_cors import CORS, cross_origin
import threading
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    flag_path = os.path.join(MODEL_DIR, "models_loaded.flag")
    models_ready = os.path.exists(flag_path)
    return jsonify({"status": "ok", "models_loaded": models_ready}), 200

class ModelManager:
    def __init__(self):
        self.loaded = False
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.keras_model = None
        self.authority_model = None
        self.bandwagon_model = None
        self.black_model = None
        self.causal_model = None
        self.slogans_model = None
        self.thought_model = None
        self.xlnet_tokenizer = None
        self.xlnet_model = None
        self.storage_client = storage.Client(project='iconic-market-452120-a0')
    
    def download_folder_from_gcs(self, bucket, prefix, local_dir):
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            relative_path = blob.name[len(prefix):].lstrip("/")
            local_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            logging.info(f"Downloading {blob.name} to {local_path}...")
            try:
                blob.download_to_filename(local_path)
            except Exception as e:
                logging.error(f"Error downloading {blob.name}: {e}")
        logging.info(f"✅ Folder '{prefix}' downloaded to '{local_dir}'.")
    
    def download_file(self, blob_name, destination_file):
        local_path = os.path.join(MODEL_DIR, destination_file)
        if not os.path.exists(local_path):
            try:
                logging.info(f"Downloading {blob_name} from GCS...")
                bucket = self.storage_client.bucket(BUCKET_NAME)
                blob = bucket.blob(blob_name)
                blob.download_to_filename(local_path)
                logging.info(f"✅ Downloaded {blob_name} to {local_path}")
            except Exception as e:
                logging.error(f"Error downloading {blob_name}: {e}")
        else:
            logging.info(f"✔ {blob_name} already exists locally.")
    
    def load_models(self):
        try:
            # List of model files to download
            model_files = [
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
            for file in model_files:
                self.download_file(file, file)
            bucket = self.storage_client.bucket(BUCKET_NAME)
            # Download folder for XLNet model only
            xlnet_prefix = "xlnet_trained_model/"  # Include trailing slash
            xlnet_local = os.path.join(MODEL_DIR, "xlnet_trained_model")
            os.makedirs(xlnet_local, exist_ok=True)
            self.download_folder_from_gcs(bucket, xlnet_prefix, xlnet_local)
            
            logging.info("🔄 Loading models...")
            self.tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
            self.label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
            
            self.keras_model = keras.models.load_model(os.path.join(MODEL_DIR, "text_classification_model.keras"))
            self.authority_model = keras.models.load_model(os.path.join(MODEL_DIR, "Appeal_to_Authority_model.keras"))
            self.bandwagon_model = keras.models.load_model(os.path.join(MODEL_DIR, "Bandwagon_Reductio_ad_hitlerum_model.keras"))
            self.black_model = keras.models.load_model(os.path.join(MODEL_DIR, "Black-and-White_Fallacy_model.keras"))
            self.causal_model = keras.models.load_model(os.path.join(MODEL_DIR, "Causal_Oversimplification_model.keras"))
            self.slogans_model = keras.models.load_model(os.path.join(MODEL_DIR, "Slogans_model.keras"))
            self.thought_model = keras.models.load_model(os.path.join(MODEL_DIR, "Thought-terminating_Cliches_model.keras"))
            
            xlnet_model_path = os.path.join(MODEL_DIR, "xlnet_trained_model")
            self.xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
            self.xlnet_model = TFAutoModelForSequenceClassification.from_pretrained(xlnet_model_path)
            
            logging.info("✅ Models loaded.")
            self.loaded = True
            # Create a flag file indicating that models are loaded.
            flag_path = os.path.join(MODEL_DIR, "models_loaded.flag")
            with open(flag_path, "w") as f:
                f.write("true")
            return True
        except Exception as e:
            logging.error("General error in load_models: " + str(e))
            self.loaded = False
            return False

# Create a global instance of ModelManager
model_manager = ModelManager()
# Start loading models in a background thread
threading.Thread(target=model_manager.load_models, daemon=True).start()

# Load spaCy model (assumed lightweight)
nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# Prediction functions using the model_manager instance
def predict_xlnet(text):
    inputs = model_manager.xlnet_tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    logits = model_manager.xlnet_model(**inputs).logits
    probs = np.array(tf.nn.softmax(logits, axis=1))
    return probs

def pad_vectorized_text(vectorized_text, expected_shape):
    current_shape = vectorized_text.shape[1]
    if current_shape < expected_shape:
        padding = np.zeros((vectorized_text.shape[0], expected_shape - current_shape))
        return np.hstack((vectorized_text, padding))
    return vectorized_text[:, :expected_shape]

def predict_keras(text):
    vectorized_text = model_manager.tfidf_vectorizer.transform([text]).toarray()
    vectorized_text = pad_vectorized_text(vectorized_text, 15396)
    probs = model_manager.keras_model.predict(vectorized_text)
    return probs

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def ensemble_multiclass_predict(text):
    p_xlnet = predict_xlnet(text)
    p_keras = predict_keras(text)
    p_xlnet = softmax(p_xlnet)
    p_keras = softmax(p_keras)
    avg_probs = (p_xlnet + p_keras) / 2
    final_prediction = np.argmax(avg_probs, axis=1)
    return final_prediction[0], avg_probs

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    # Check if the flag file exists.
    flag_path = os.path.join(MODEL_DIR, "models_loaded.flag")
    if not os.path.exists(flag_path):
        return jsonify({"error": "Models are still loading, please try again later."}), 503

    # Also check that critical attributes are not None.
    if model_manager.xlnet_tokenizer is None or model_manager.xlnet_model is None:
        return jsonify({"error": "xln models are still loading, please try again later."}), 503

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400
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
    except Exception as e:
        logging.error("Error in /predict: " + str(e))
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info("Starting server on port: " + str(port))
    app.run(host='0.0.0.0', port=port)

