# Импорт библиотек
import spacy
import tensorflow as tf
import numpy as np
from transformers import XLNetTokenizer, TFAutoModelForSequenceClassification, RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from flask import Flask, request, jsonify
from google.colab import drive
import os
from dotenv import load_dotenv

#health check
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# Загружаем переменные из .env
load_dotenv()

# Теперь можно использовать os.getenv() для доступа к переменным
XLNET_MODEL_PATH = os.getenv("XLNET_MODEL_PATH")
ROBERTA_MODEL_PATH = os.getenv("ROBERTA_MODEL_PATH")
KERAS_MODEL_PATH = os.getenv("KERAS_MODEL_PATH")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH")
TFIDF_VECTORIZER_PATH = os.getenv("TFIDF_VECTORIZER_PATH")

APPEAL_TO_AUTHORITY_MODEL_PATH = os.getenv("APPEAL_TO_AUTHORITY_MODEL_PATH")
BANDWAGON_MODEL_PATH = os.getenv("BANDWAGON_MODEL_PATH")
BLACK_WHITE_FALLACY_MODEL_PATH = os.getenv("BLACK_WHITE_FALLACY_MODEL_PATH")
CAUSAL_OVERSIMPLIFICATION_MODEL_PATH = os.getenv("CAUSAL_OVERSIMPLIFICATION_MODEL_PATH")
SLOGANS_MODEL_PATH = os.getenv("SLOGANS_MODEL_PATH")
THOUGHT_TERMINATING_CLICHES_MODEL_PATH = os.getenv("THOUGHT_TERMINATING_CLICHES_MODEL_PATH")

# Подключение Google Drive (если модели хранятся там)
drive.mount('/content/drive')

# =========================
# 1️⃣ Настройка spaCy для разбиения текста
# =========================
nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    """Разбивает текст на предложения с помощью spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# =========================
# 2️⃣ Загрузка обученных моделей
# =========================

# Загрузка XLNet
xlnet_model_path = '/content/drive/MyDrive/xlnet_trained_model2'
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = TFAutoModelForSequenceClassification.from_pretrained(xlnet_model_path)

# Загрузка RoBERTa
roberta_model_path = '/content/drive/MyDrive/roberta_trained_model'
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = TFRobertaForSequenceClassification.from_pretrained(roberta_model_path, from_pt=True)

# Загрузка TensorFlow.keras модели
keras_model_path = '/content/drive/MyDrive/text_classification_model.keras'
keras_model = tf.keras.models.load_model(keras_model_path)

# Загрузка TF-IDF векторизатора
tfidf_path = '/content/drive/MyDrive/vectorizer.joblib'
tfidf_vectorizer = joblib.load(tfidf_path)

# =========================
# 3️⃣ Функции для предсказаний
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
# 4️⃣ Запуск Flask API
# =========================
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """API-метод для предсказания."""
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Разбиваем текст на предложения с помощью spaCy
    sentences = split_sentences(text)

    results = []
    for sentence in sentences:
        multiclass_prediction, multiclass_probs = ensemble_multiclass_predict(sentence)
        
        results.append({
            "sentence": sentence,
            "Multiclass Prediction": multiclass_prediction,
        })

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
