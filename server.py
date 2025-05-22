'''
# server.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging
import urllib.request

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import spacy
import numpy as np
import joblib

import torch
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetForSequenceClassification

import keras
from keras.layers import InputLayer
from keras.models import load_model

from huggingface_hub import hf_hub_download


# ========== Настройки ==========
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# ========== Патч для InputLayer ==========
class CustomInputLayer(InputLayer):
    def __init__(self, **kwargs):
        batch_shape = kwargs.pop('batch_shape', None)
        if batch_shape is not None:
            # Убираем batch_shape и ставим input_shape = (dims...,)
            kwargs['input_shape'] = tuple(batch_shape[1:])
        super().__init__(**kwargs)

MODELS = None

# ========== Предзагрузка моделей ==========
def init_models():
    logging.info("⏳ Загрузка моделей…")

    # 1) Скачиваем все `.keras` и статические файлы из HF
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
    local = {}
    for fname, repo in hf_repos.items():
        path = hf_hub_download(repo_id=repo, filename=fname, cache_dir=MODEL_DIR, repo_type="model")
        local[fname] = path
        logging.info(f"✅ {fname} скачан в {path}")

    models = {}
    # 2) TF-IDF + LabelEncoder
    models['tfidf'] = joblib.load(local["vectorizer.joblib"])
    models['le']    = joblib.load(local["label_encoder.joblib"])

    # 3) Keras-модели
    def load_keras(key):
        return load_model(
            local[key],
            custom_objects={
                'Functional': keras.models.Model,
                'InputLayer': CustomInputLayer
            },
            compile=False
        )

    models['mc_keras']    = load_keras("text_classification_model.keras")
    models['bin_auth']    = load_keras("Appeal_to_Authority_model.keras")
    models['bin_band']    = load_keras("Bandwagon_Reductio_ad_hitlerum_model.keras")
    models['bin_bwfall']  = load_keras("Black-and-White_Fallacy_model.keras")
    models['bin_causal']  = load_keras("Causal_Oversimplification_model.keras")
    models['bin_slog']    = load_keras("Slogans_model.keras")
    models['bin_thou']    = load_keras("Thought-terminating_Cliches_model.keras")

    # 4) XLNet через PyTorch
    models['xlnet_tok'] = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    models['xlnet_mc']  = XLNetForSequenceClassification.from_pretrained("brsvaaa/xlnet_trained_model")

    # 5) spaCy
    models['nlp'] = spacy.load("en_core_web_sm")

    logging.info("✅ Все модели загружены успешно.")
    return models

def get_models():
    global MODELS
    if MODELS is None:
        MODELS = init_models()
    return MODELS

# ========== Утилиты предсказания ==========
def split_sentences(text: str):
    nlp = get_models()['nlp']
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def predict_xlnet(text: str):
    m = get_models()
    tok = m['xlnet_tok'](text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    tok = {k: v.to(m['xlnet_mc'].device) for k, v in tok.items()}
    with torch.no_grad():
        logits = m['xlnet_mc'](**tok).logits
    return F.softmax(logits, dim=1).cpu().numpy()

def pad_to_expected(x: np.ndarray, target_dim: int):
    current = x.shape[1]
    if current < target_dim:
        return np.hstack([x, np.zeros((x.shape[0], target_dim-current))])
    return x[:, :target_dim]
    
def predict_keras_mc(text: str):
    m = get_models()
    vec = m['tfidf'].transform([text]).toarray()
    expected = m['mc_keras'].input_shape[-1]
    vec = pad_to_expected(vec, expected)
    return m['mc_keras'].predict(vec)

def predict_binary_label(text: str):
    m = get_models()
    # Сначала получаем «сырой» вектор TF-IDF
    raw_vec = m['tfidf'].transform([text]).toarray()

    # Перечень бинарных моделей + их индексов
    binary_models = [
        ('bin_auth',   0),   # Appeal to Authority
        ('bin_band',   2),   # Bandwagon / Reductio ad Hitlerum
        ('bin_bwfall', 3),   # Black-and-White Fallacy
        ('bin_causal', 4),   # Causal Oversimplification
        ('bin_slog',  12),   # Slogans
        ('bin_thou',  13),   # Thought-terminating Cliches
    ]

    for model_key, label_idx in binary_models:
        model = m[model_key]
        # Выясняем, какой размер входного слоя у этой бинарки
        expected = model.input_shape[-1]
        # Подгоняем наш raw_vec под эту длину
        if raw_vec.shape[1] < expected:
            vec = np.hstack([raw_vec, np.zeros((raw_vec.shape[0], expected - raw_vec.shape[1]))])
        else:
            vec = raw_vec[:, :expected]

        # Делаем предсказание
        prob = model.predict(vec)
        # Интерпретируем выход (два нейрона? или один?)
        if prob.ndim == 2 and prob.shape[1] == 2:
            score = float(prob[0, 1])
        else:
            score = float(prob[0, 0])
        if score > 0.5:
            return label_idx

    return None
    
def ensemble_multiclass_predict(text: str):
    bin_idx = predict_binary_label(text)
    if bin_idx is not None:
        # Если бинарная модель "нашла" технику — сразу её и возвращаем
        return bin_idx, None
    
    p1 = predict_xlnet(text)
    p2 = predict_keras_mc(text)
    avg = (p1 + p2) / 2
    cls = int(np.argmax(avg, axis=1)[0])
    return cls, avg


# ========== Flask-эндпоинты ==========
@app.route('/', methods=['GET'])
def index():
    # For a simple JSON response:
    return jsonify({
        "service": "Propaganda Detector API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status="ok"), 200

@app.route('/predict', methods=['POST','OPTIONS'])
@cross_origin(origins='*')
def predict():
    if request.method == 'OPTIONS':
        return jsonify(message='OK'), 200

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify(error="No text provided"), 400

    sentences = split_sentences(text)
    results = []
    for s in sentences:
        cls, _ = ensemble_multiclass_predict(s)
        results.append({
            "sentence": s,
            "Multiclass_Prediction": cls
        })

    return jsonify(results=results), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logging.info(f"Запуск на порту {port}")
    app.run(host='0.0.0.0', port=port)

'''
# server.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging
import urllib.request

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import spacy
import numpy as np
import joblib

import torch
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetForSequenceClassification

import keras
from keras.layers import InputLayer
from keras.models import load_model

from huggingface_hub import hf_hub_download
import tensorflow as tf  # CHANGED: добавлен импорт tensorflow


# ========== Настройки ==========
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# ========== Патч для InputLayer ==========
class CustomInputLayer(InputLayer):
    def __init__(self, **kwargs):
        batch_shape = kwargs.pop('batch_shape', None)
        if batch_shape is not None:
            # Убираем batch_shape и ставим input_shape = (dims...,)
            kwargs['input_shape'] = tuple(batch_shape[1:])
        super().__init__(**kwargs)

MODELS = None

# ========== Предзагрузка моделей ==========
def init_models():
    logging.info("⏳ Загрузка моделей…")

    # 1) Скачиваем все `.keras` и статические файлы из HF
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
    local = {}
    for fname, repo in hf_repos.items():
        path = hf_hub_download(repo_id=repo, filename=fname, cache_dir=MODEL_DIR, repo_type="model")
        local[fname] = path
        logging.info(f"✅ {fname} скачан в {path}")

    models = {}
    # 2) TF-IDF + LabelEncoder
    models['tfidf'] = joblib.load(local["vectorizer.joblib"])
    models['le']    = joblib.load(local["label_encoder.joblib"])

    # 3) Keras-модели
    def load_keras(key):
        # CHANGED: сначала сохраняем в переменную, не возвращаем сразу
        model = load_model(
            local[key],
            custom_objects={
                'Functional': keras.models.Model,
                'InputLayer': CustomInputLayer
            },
            compile=False
        )
        # CHANGED: определяем свой tf.function только вокруг чистого вызова model.call
        @tf.function(
            input_signature=[tf.TensorSpec([None, model.input_shape[-1]], tf.float32)],
            reduce_retracing=True
        )
        def infer(x):
            return model(x, training=False)
 
        model._inference_fn = infer
        model.make_predict_function()  # на всякий случай
        return model
        
    # сразу загружаем все Keras-модели через одну функцию
    models['mc_keras']    = load_keras("text_classification_model.keras")
    models['bin_auth']    = load_keras("Appeal_to_Authority_model.keras")
    models['bin_band']    = load_keras("Bandwagon_Reductio_ad_hitlerum_model.keras")
    models['bin_bwfall']  = load_keras("Black-and-White_Fallacy_model.keras")
    models['bin_causal']  = load_keras("Causal_Oversimplification_model.keras")
    models['bin_slog']    = load_keras("Slogans_model.keras")
    models['bin_thou']    = load_keras("Thought-terminating_Cliches_model.keras")

    
    # 4) XLNet через PyTorch
    models['xlnet_tok'] = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    models['xlnet_mc']  = XLNetForSequenceClassification.from_pretrained("brsvaaa/xlnet_trained_model")

    # 5) spaCy
    models['nlp'] = spacy.load("en_core_web_sm")

    logging.info("✅ Все модели загружены успешно.")
    return models

def get_models():
    global MODELS
    if MODELS is None:
        MODELS = init_models()
    return MODELS

# ========== Утилиты предсказания ==========
def split_sentences(text: str):
    nlp = get_models()['nlp']
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def predict_xlnet_batch(sentences):
    m = get_models()
    # 1) Токенизируем сразу все предложения
    batch = m['xlnet_tok'](
        sentences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    # 2) Переносим на устройство
    batch = {k: v.to(m['xlnet_mc'].device) for k,v in batch.items()}
    # 3) Делаем один forward
    with torch.no_grad():
        logits = m['xlnet_mc'](**batch).logits  # shape (N, C)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs  # numpy array (N, C)

def pad_to_expected(x: np.ndarray, target_dim: int):
    current = x.shape[1]
    if current < target_dim:
        return np.hstack([x, np.zeros((x.shape[0], target_dim-current))])
    return x[:, :target_dim]
    
def predict_keras_batch(sentences):
    m = get_models()
    # 1) TF-IDF сразу на весь список
    X = m['tfidf'].transform(sentences).toarray()  # shape (N, D0)
    D1 = m['mc_keras'].input_shape[-1]
    # 2) pad/truncate всё сразу
    if X.shape[1] < D1:
        X = np.hstack([X, np.zeros((X.shape[0], D1-X.shape[1]))])
    else:
        X = X[:, :D1]

    X = X.astype(np.float32)
    # CHANGED: вызываем нашу обёртку, затем .numpy()
    preds = m['mc_keras']._inference_fn(tf.constant(X, dtype=tf.float32))
    return preds.numpy()

def predict_binary_batch(sentences):
    m = get_models()
    raw = m['tfidf'].transform(sentences).toarray()  # shape (N, D0)
    results = [None]*len(sentences)
    # для каждой бинарной модели
    for model_key, label_idx in [
        ('bin_auth',   0),
        ('bin_band',   2),
        ('bin_bwfall', 3),
        ('bin_causal', 4),
        ('bin_slog',  12),
        ('bin_thou',  13),
    ]:
        model = m[model_key]
        D_bin = model.input_shape[-1]
        X = raw
        if raw.shape[1] < D_bin:
            X = np.hstack([raw, np.zeros((raw.shape[0], D_bin-raw.shape[1]))])
        else:
            X = raw[:, :D_bin]

        X = X.astype(np.float32)
        probs = model._inference_fn(tf.constant(X, dtype=tf.float32)).numpy()        # извлечём вероятность класса «1» для всех N
        if probs.ndim==2 and probs.shape[1]==2:
            scores = probs[:,1]
        else:
            scores = probs[:,0]
        for i, s in enumerate(scores):
            if s>0.5 and results[i] is None:
                results[i] = label_idx
    return results  # list длины N, с индексами или None
    
def ensemble_batch(sentences):
    N = len(sentences)
    bin_labels = predict_binary_batch(sentences)        # [idx_or_None]*N
    mc1 = predict_xlnet_batch(sentences)               # (N,C)
    mc2 = predict_keras_batch(sentences)               # (N,C)
    avg = (mc1 + mc2) / 2.0                            # (N,C)
    final = []
    for i in range(N):
        if bin_labels[i] is not None:
            final.append(bin_labels[i])
        else:
            final.append(int(np.argmax(avg[i])))
    return final


# ========== Flask-эндпоинты ==========
@app.route('/', methods=['GET'])
def index():
    # For a simple JSON response:
    return jsonify({
        "service": "Propaganda Detector API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status="ok"), 200

@app.route('/predict', methods=['POST','OPTIONS'])
@cross_origin(origins='*')
def predict():
    if request.method=='OPTIONS':
        return jsonify(message='OK'), 200

    data = request.get_json(silent=True) or {}
    text = data.get('text','').strip()
    if not text:
        return jsonify(error="No text provided"), 400

    sents = split_sentences(text)    # list of sentences
    labels = ensemble_batch(sents)   # list of ints, длины len(sents)

    results = [
        {"sentence": sents[i], "Multiclass_Prediction": labels[i]}
        for i in range(len(sents))
    ]
    return jsonify(results=results), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logging.info(f"Запуск на порту {port}")
    app.run(host='0.0.0.0', port=port)


