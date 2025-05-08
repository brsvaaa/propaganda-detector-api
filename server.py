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
MODEL_DIR = "models"  # Local directory to store downloaded models
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import tf_keras as keras
import numpy as np
from transformers import XLNetTokenizer, TFAutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from flask import Flask, request, jsonify, g
from flask_cors import CORS, cross_origin
import threading
import logging
import urllib

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "models_loaded": "models" in g}), 200

def download_from_huggingface(repo_url, filename):
    dest_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(dest_path):
        try:
            logging.info(f"Downloading {filename} from Hugging Face...")
            urllib.request.urlretrieve(f"{repo_url}/{filename}", dest_path)
            logging.info(f"✅ Downloaded {filename} to {dest_path}")
        except Exception as e:
            logging.error(f"Error downloading {filename} from HF: {e}")
    else:
        logging.info(f"✔ {filename} already exists locally.")

def init_models():
    from huggingface_hub import hf_hub_download
    try:
        # В новых версиях может лежать в utils
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        HfHubHTTPError = Exception

    models = {}
    local_paths = {}
    # Mapping filenames to their Hugging Face repo_ids
    hf_repos = {
        "Appeal_to_Authority_model.keras": "brsvaaa/Appeal_to_Authority_model.keras",
        "Bandwagon_Reductio_ad_hitlerum_model.keras": "brsvaaa/Bandwagon_Reductio_ad_hitlerum_model.keras",
        "Black-and-White_Fallacy_model.keras": "brsvaaa/Black-and-White_Fallacy_model.keras",
        "Causal_Oversimplification_model.keras": "brsvaaa/Causal_Oversimplification_model.keras",
        "Slogans_model.keras": "brsvaaa/Slogans_model.keras",
        "Thought-terminating_Cliches_model.keras": "brsvaaa/Thought-terminating_Cliches_model.keras",
        "text_classification_model.keras": "brsvaaa/text_classification_model.keras",
        "vectorizer.joblib": "brsvaaa/vectorizer.joblib",
        "label_encoder.joblib": "brsvaaa/label_encoder.joblib"
    }
    
    for filename, repo_id in hf_repos.items():
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=MODEL_DIR,
                repo_type="model"
            )
            local_paths[filename] = path
            logging.info(f"✅ {filename} скачан в {path}")
        except HfHubHTTPError as e:
            logging.error(f"❌ HTTP-ошибка при скачивании {filename} из {repo_id}: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Ошибка при скачивании {filename} из {repo_id}: {e}")
            raise
    # 2) Загружаем vectorizer и encoder по реальным путям
    models['vectorizer.joblib'] = joblib.load(local_paths["vectorizer.joblib"])
    models['label_encoder']  = joblib.load(local_paths["label_encoder.joblib"])

    # 3) Загружаем Keras-модели
    def load_keras_model_from(path_key):
        p = local_paths[path_key]
        return keras.models.load_model(
            p,
            custom_objects={
                # Класс Functional внутри вашего .keras файла
                "Functional": tf.keras.Model,
                # Подпомогаем InputLayer
                "InputLayer": tf.keras.layers.InputLayer
            },
            compile=False
        )

    models['keras_model']     = load_keras_model_from("text_classification_model.keras")
    models['authority_model'] = load_keras_model_from("Appeal_to_Authority_model.keras")
    models['bandwagon_model'] = load_keras_model_from("Bandwagon_Reductio_ad_hitlerum_model.keras")
    models['black_model']     = load_keras_model_from("Black-and-White_Fallacy_model.keras")
    models['causal_model']    = load_keras_model_from("Causal_Oversimplification_model.keras")
    models['slogans_model']   = load_keras_model_from("Slogans_model.keras")
    models['thought_model']   = load_keras_model_from("Thought-terminating_Cliches_model.keras")

    # 4) Загружаем XLNet-трансформер
    models['xlnet_tokenizer'] = XLNetTokenizer.from_pretrained("brsvaaa/xlnet_trained_model")
    models['xlnet_model']     = TFAutoModelForSequenceClassification.from_pretrained(
        "brsvaaa/xlnet_trained_model",
        cache_dir=MODEL_DIR
    )

    logging.info("✅ Все модели загружены!")
    return models

def get_models():
    if 'models' not in g:
        logging.info("Models not in flask.g; loading now...")
        g.models = init_models()
    return g.models
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
    inputs = models['xlnet_tokenizer'](text, return_tensors="tf", truncation=True, padding=True)
    logits = models['xlnet_model'](**inputs).logits
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
    vectorized_text = models['vectorizer.joblib'].transform([text]).toarray()
    vectorized_text = pad_vectorized_text(vectorized_text, 15396)
    probs = models['keras_model'].predict(vectorized_text)
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

    try:
        
        models = get_models() 
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

