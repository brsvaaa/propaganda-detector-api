
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
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from flask import send_from_directory
from huggingface_hub import hf_hub_download
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
torch.set_num_threads(1)


torch.set_num_threads(1)

CONF_THRESHOLD = 0.3


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)


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


def init_models():
    logging.info("⏳ Загрузка моделей…")

    hf_repos = {
        "multi_binary.keras": "brsvaaa/multi_binary.keras",
        "text_classification_model.keras": "brsvaaa/text_classification_model.keras",
        "vectorizer.joblib": "brsvaaa/vectorizer.joblib",
        "label_encoder.joblib": "brsvaaa/label_encoder.joblib"
    }
    local = {}
    for fname, repo in hf_repos.items():
        try:
            path = hf_hub_download(repo_id=repo, filename=fname, cache_dir=MODEL_DIR, repo_type="model")
            local[fname] = path
            logging.info(f"✅ {fname} скачан")
        except Exception as e:
            logging.error(f"❌ Не удалось скачать {fname}: {e}")
            # можно либо аварийно завершить, либо возвратить специальный флаг
            raise RuntimeError(f"Ошибка загрузки моделей: {e}")


    models = {}
    #  TF-IDF + LabelEncoder
    models['tfidf'] = joblib.load(local["vectorizer.joblib"])
    models['le']    = joblib.load(local["label_encoder.joblib"])

    #  Keras-модели
    models['mc_keras'] = load_model(
        local["text_classification_model.keras"],
        custom_objects={
            'Functional': keras.models.Model,
            'InputLayer': CustomInputLayer
        },
        compile=False
    )

    models['multi_binary'] = load_model(
        local["multi_binary.keras"],
        custom_objects={
            'CustomInputLayer': CustomInputLayer,
            'SliceLayer':        SliceLayer,
            'PickProbLayer':     PickProbLayer,
            'Functional':        keras.models.Model,
        },
        compile=False
    )

    
    #  XLNet через PyTorch
    models['xlnet_tok'] = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    models['xlnet_mc']  = XLNetForSequenceClassification.from_pretrained("brsvaaa/xlnet_trained_model")
    models['xlnet_mc'].eval()
    if torch.cuda.is_available():
        models['xlnet_mc'].half()

    #  spaCy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    models['nlp'] = nlp
    
    logging.info("✅ Multi-output binary model built.")
    return models



def get_models():
    global MODELS
    if MODELS is None:
        MODELS = init_models()
    return MODELS


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
    chunk_size = 10

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


