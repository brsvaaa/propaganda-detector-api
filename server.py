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
import tensorflow as tf  
import keras
from keras.layers import InputLayer
from keras.models import load_model
from huggingface_hub import hf_hub_download

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

MODELS = None

class CustomInputLayer(InputLayer):
    def __init__(self, **kwargs):
        batch_shape = kwargs.pop('batch_shape', None)
        if batch_shape is not None:
            # Убираем batch_shape и ставим input_shape = (dims...,)
            kwargs['input_shape'] = tuple(batch_shape[1:])
        super().__init__(**kwargs)

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
binary_models = {}
# 2) TF-IDF + LabelEncoder
models['vectorizer'] = joblib.load(local["vectorizer.joblib"])
models['label_encoder']  = joblib.load(local["label_encoder.joblib"])
label_encoder = models['label_encoder']
class_labels  = list(label_encoder.classes_)


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
binary_models = [
    load_keras("Appeal_to_Authority_model.keras"),
    load_keras("Bandwagon_Reductio_ad_hitlerum_model.keras"),
    load_keras("Black-and-White_Fallacy_model.keras"),
    load_keras("Causal_Oversimplification_model.keras"),
    load_keras("Slogans_model.keras"),
    load_keras("Thought-terminating_Cliches_model.keras"),
]
binary_model_names = [
    "Appeal_to_Authority",
    "Bandwagon_Reductio_ad_hitlerum",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Slogans",
    "Thought-terminating_Cliches",
]
# 4) XLNet через PyTorch
models['xlnet_tok'] = XLNetTokenizer.from_pretrained("xlnet-base-cased")
models['xlnet_mc']  = XLNetForSequenceClassification.from_pretrained("brsvaaa/xlnet_trained_model")
models['xlnet_mc'].eval()  # >>> ADDED: устанавливаем режим инференса для PyTorch модели

xlnet_tokenizer = models['xlnet_tok']
xlnet_model     = models['xlnet_mc']
# 5) spaCy
models['nlp'] = spacy.load("en_core_web_sm")

logging.info("✅ Все модели загружены успешно.")



# Установка порога уверенности для бинарных моделей
CONF_THRESHOLD = 0.8

# Максимальная длина последовательности (должна соответствовать обучению моделей)
MAX_SEQ_LEN = 512  # например, 128 токенов; установить фактически используемое значение

def pad_to_expected(x: np.ndarray, target_dim: int):
    current = x.shape[1]
    if current < target_dim:
        return np.hstack([x, np.zeros((x.shape[0], target_dim-current))])
    return x[:, :target_dim]
# >>> ADDED: Компиляция (прогрев) моделей на холостом примере, чтобы избежать retracing:contentReference[oaicite:5]{index=5}
# Создаём фиктивный батч данных для прогрева Keras моделей
dummy_tfidf = models['vectorizer'].transform([""]).toarray()
mc_expected = models['mc_keras'].input_shape[-1]
mc_dummy = pad_to_expected(dummy_tfidf, mc_expected).astype(np.float32)
# прогрев через predict_on_batch (или через ваш tf.function infer_fn)
_ = models['mc_keras'].predict_on_batch(mc_dummy)

for bin_model in binary_models:
    # одинаково «прогреваем» через predict_on_batch
    bin_expected = bin_model.input_shape[-1]
    bin_dummy = pad_to_expected(dummy_tfidf, bin_expected).astype(np.float32)
    _ = bin_model.predict_on_batch(bin_dummy)

# Обёртки tf.function для Keras-моделей (чтобы выполнять инференс в графе, без retrace)
# >>> ADDED: Определяем tf.function один раз вне цикла запросов:contentReference[oaicite:6]{index=6}
@tf.function
def keras_binary_batch_predict(model, batch):
    return model(batch, training=False)

@tf.function
def keras_multi_batch_predict(batch):
    return models['mc_keras'](batch, training=False)


# Функция предобработки текста: разбивает статью на предложения
def split_to_sentences(text):
    # >>> CHANGED: Используем простой способ разбиения на предложения.
    # Можно заменить на более точный (nltk.sent_tokenize и т.п.) при необходимости.
    import re
    # Разбиваем по точкам, восклицательным и вопросительным знакам, сохраняя разделители
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s for s in sentences if s]  # убираем пустые строки


    
# Функция токенизации для Keras моделей (например, на основе простого Tokenizer или словаря)
# (Предполагается, что модели ожидают на вход последовательности токенов фиксированной длины MAX_SEQ_LEN)
def tokenize_and_pad(sentences):
    m = models  # CHANGED: берём TF-IDF векторизацию из глобальных моделей
    
    # CHANGED: векторизуем весь батч предложений
    vecs = m['vectorizer'].transform(sentences).toarray()
    
    # CHANGED: подгоняем/усекаем до требуемой длины Keras-модели
    expected = m['mc_keras'].input_shape[-1]
    padded = pad_to_expected(vecs, expected)
    
    return padded

# Основная функция для классификации списка предложений
def classify_sentences(sentences):
    results = []
    # Подготовим батч для всех предложений сразу (для Keras моделей)
    X = tokenize_and_pad(sentences)
    X = X.astype(np.int32)  # стандартизируем тип для Keras (int32 для embedding)
    num_sent = X.shape[0]

    # >>> CHANGED: Единовременный батчевый запуск всех бинарных моделей:contentReference[oaicite:7]{index=7}
    binary_preds = [
        np.array(model.predict_on_batch(X.astype(np.float32)))
        for model in binary_models
    ]    

    binary_preds = np.stack(binary_preds, axis=1).squeeze()  
    # Теперь binary_preds имеет форму (num_sent, num_binary_models) с вероятностями по каждому бинарному классификатору

    # Выбираем для каждого предложения, сработал ли какой-то бинарный классификатор уверенно
    # Инициализируем список итоговых меток None (для предложений, которые пойдут на мультиклассовую модель)
    final_labels = [None] * num_sent

    for i, probs in enumerate(binary_preds):
        max_idx  = int(np.argmax(probs))
        max_prob = float(probs[max_idx])
        if max_prob >= CONF_THRESHOLD:
            final_labels[i] = binary_model_names[max_idx]

    # Собираем предложения, которым требуется мультиклассовая классификация
    to_multi_idxs = [idx for idx, label in enumerate(final_labels) if label is None]
    if to_multi_idxs:
        to_multi_sentences = [sentences[idx] for idx in to_multi_idxs]
        # Токенизация для этих предложений (можно переиспользовать X, но для простоты отдельно)
        X_multi = tokenize_and_pad(to_multi_sentences)
        X_multi = X_multi.astype(np.int32)
        batch_size = X_multi.shape[0]

        # Keras мультиклассовые предсказания (батч)
        multi_preds_keras = np.array(keras_multi_batch_predict(X_multi))  # форма (batch_size, num_classes)

        # PyTorch XLNet мультиклассовые предсказания (батч)
        encodings = xlnet_tokenizer(to_multi_sentences,
                                     return_tensors='pt',
                                     padding=True,
                                     truncation=True,
                                     max_length=MAX_SEQ_LEN)
        with torch.no_grad():
            outputs = xlnet_model(**encodings)
        logits = outputs.logits  # тензор размера (batch_size, num_classes)
        multi_preds_torch = torch.softmax(logits, dim=1).cpu().numpy()

        # Усредняем прогнозы двух моделей для более устойчивого результата
        combined_preds = (multi_preds_keras + multi_preds_torch) / 2.0
        # Выбираем наиболее вероятный класс для каждого предложения
        pred_classes = combined_preds.argmax(axis=1)

        # Записываем результаты в соответствующие позиции final_labels
        for j, idx in enumerate(to_multi_idxs):
            class_idx = int(pred_classes[j])
            # Получаем название класса (техники) по индексу
            label_name = class_labels[class_idx] if label_encoder else str(class_idx)
            final_labels[idx] = label_name

    return final_labels

# Flask маршрут для предсказания
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

@app.route('/predict', methods=['POST'])
@cross_origin(origins='*')
def predict():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "No input text provided"}), 400

    text = data['text']
    # Разбиваем текст статьи на отдельные предложения
    sentences = split_to_sentences(text)
    if not sentences:
        return jsonify({"results": []})

    # >>> CHANGED: Классифицируем предложения батчами, используя оптимизированную функцию
    predictions = classify_sentences(sentences)

    # Формируем результат: список словарей с предложением и предсказанной меткой
    results = []
    for sent, label in zip(sentences, predictions):
        results.append({"sentence": sent, "label": label})

    return jsonify({"results": results})

# Запуск приложения
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

import psutil

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
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from flask import send_from_directory
from huggingface_hub import hf_hub_download
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# PyTorch тоже ограничит число потоков
torch.set_num_threads(1)

CONF_THRESHOLD = 0.5

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

class SliceLayer(Layer):
    def __init__(self, d, **kwargs):
        super().__init__(**kwargs)
        self.d = d

    def call(self, inputs):
        return inputs[:, :self.d]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'d': self.d})
        return cfg

class PickProbLayer(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return tf.expand_dims(inputs[:, self.index], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'index': self.index})
        return cfg
BINARY_LABELS = [0, 2, 3, 4, 12, 13]
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
    models['mc_keras'] = load_model(
        local["text_classification_model.keras"],
        custom_objects={
            'Functional': keras.models.Model,
            'InputLayer': CustomInputLayer
        },
        compile=False
    )

    
    # 4) XLNet через PyTorch
    models['xlnet_tok'] = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    models['xlnet_mc']  = XLNetForSequenceClassification.from_pretrained("brsvaaa/xlnet_trained_model")
    models['xlnet_mc'].eval()
    if torch.cuda.is_available():
        models['xlnet_mc'].half()

    # 5) spaCy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    models['nlp'] = nlp

    submodels = []
    for key in [
        "Appeal_to_Authority_model.keras",
        "Bandwagon_Reductio_ad_hitlerum_model.keras",
        "Black-and-White_Fallacy_model.keras",
        "Causal_Oversimplification_model.keras",
        "Slogans_model.keras",
        "Thought-terminating_Cliches_model.keras",
    ]:
        sm = load_model(
            local[key],
            custom_objects={'InputLayer': CustomInputLayer},
            compile=False
        )
        sm.trainable = False
        submodels.append(sm)

    # строим единый multi_binary
    D0 = m['tfidf'].transform([""]).shape[1]
    inp = Input(shape=(D0,), dtype=tf.float32, name="tfidf_input")
    probs = []
    for sm, label in zip(submodels, BINARY_LABELS):
        D_bin = sm.input_shape[-1]
        x_bin = SliceLayer(D_bin, name=f"{sm.name}_slice")(inp)
        out   = sm(x_bin, training=False)
        idx   = 1 if out.shape[-1] == 2 else 0
        p1    = PickProbLayer(idx, name=f"{sm.name}_p1")(out)
        probs.append(p1)

    multi_binary = Concatenate(axis=1, name="binary_probs")(probs)
    m['multi_binary'] = Model(inputs=inp, outputs=multi_binary, name="multi_binary")
    logging.info("✅ Multi-output binary model built.")
    return m



def get_models():
    global MODELS
    if MODELS is None:
        MODELS = init_models()
    return MODELS

# ========== Утилиты предсказания ==========
def split_sentences(text: str):
    return [s.text.strip() for s in get_models()['nlp'](text).sents]

def pad_to_expected(x: np.ndarray, target_dim: int):
    cur = x.shape[1]
    if cur < target_dim:
        return np.hstack([x, np.zeros((x.shape[0], target_dim-cur))])
    return x[:, :target_dim]

def predict_binary_batch(sentences):
    m = get_models()
    raw = m['tfidf'].transform(sentences).toarray().astype(np.float32)
    probs = m['multi_binary'].predict_on_batch(raw)  # (N,6)
    labels = [None]*len(sentences)
    for i,row in enumerate(probs):
        j = int(np.argmax(row))
        if row[j] > CONF_THRESHOLD:
            labels[i] = BINARY_LABELS[j]
    return labels

def predict_xlnet_batch(sentences):
    m = get_models()
    batch = m['xlnet_tok'](
        sentences, return_tensors="pt", truncation=True,
        padding=True, max_length=512
    )
    batch = {k:v.to(m['xlnet_mc'].device) for k,v in batch.items()}
    if next(m['xlnet_mc'].parameters()).dtype == torch.float16:
        for k in batch:
            batch[k] = batch[k].half()
    with torch.no_grad():
        logits = m['xlnet_mc'](**batch).logits
    return F.softmax(logits, dim=1).cpu().numpy()

def predict_keras_batch(sentences):
    m = get_models()
    raw = m['tfidf'].transform(sentences).toarray()
    D1  = m['mc_keras'].input_shape[-1]
    X   = pad_to_expected(raw, D1).astype(np.float32)
    return m['mc_keras'].predict_on_batch(X)


# ========== Flask-эндпоинты ==========
@app.route('/download/multi_binary.keras', methods=['GET'])
def download_multi_binary():
    return send_from_directory('models', 'multi_binary.keras', as_attachment=True)
    
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
    text = data.get("text","").strip()
    if not text:
        return jsonify(error="No text provided"), 400

    sentences = split_sentences(text)
    results = []
    chunk_size = 50

    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        bin_lbls = predict_binary_batch(chunk)
        to_multi = [j for j, lbl in enumerate(bin_lbls) if lbl is None]
        multi_lbls = [None]*len(chunk)

        if to_multi:
            subs = [chunk[j] for j in to_multi]
            xl = predict_xlnet_batch(subs)
            kr = predict_keras_batch(subs)
            avg = (xl + kr)/2
            preds = np.argmax(avg, axis=1)
            for idx, cl in zip(to_multi, preds):
                multi_lbls[idx] = int(cl)

        for sent, b, m in zip(chunk, bin_lbls, multi_lbls):
            results.append({
                "sentence": sent,
                "Multiclass_Prediction": b if b is not None else m
            })

    return jsonify(results=results), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logging.info(f"Запуск на порту {port}")
    app.run(host='0.0.0.0', port=port)
