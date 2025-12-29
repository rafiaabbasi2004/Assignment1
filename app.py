from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import pickle
import io
import os

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = 'action_recognition_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_LENGTH = 9 

# --- ASSET LOADING ---
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ System: Model Loaded")
except Exception as e:
    print(f"❌ System: Model Load Fail: {e}")

try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ System: Tokenizer Loaded")
except Exception as e:
    print(f"❌ System: Tokenizer Load Fail: {e}")

# Feature Extractor
base_model = tf.keras.applications.InceptionV3(weights='imagenet')
encoder = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def generate_caption(image_bytes):
    img = load_img(io.BytesIO(image_bytes), target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    feature = encoder.predict(img, verbose=0)
    feature = np.reshape(feature, (1, 2048))
    
    in_text = 'startseq'
    probs = []
    
    for i in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        
        yhat = model.predict([feature, sequence], verbose=0)
        
        # FIXED: convert float32 to standard Python float
        current_prob = float(np.max(yhat)) 
        probs.append(current_prob)
        
        word_idx = np.argmax(yhat)
        word = tokenizer.index_word.get(int(word_idx)) # Ensure int
        
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
        
    caption = in_text.replace('startseq', '').strip()
    
    # Calculate average and round
    avg_confidence = (sum(probs) / len(probs)) * 100 if probs else 0
    return caption, round(float(avg_confidence), 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file_bytes = request.files['file'].read()
    try:
        caption, confidence = generate_caption(file_bytes)
        # JSON standard only accepts standard floats, not numpy float32
        return jsonify({
            'caption': str(caption), 
            'confidence': float(confidence)
        })
    except Exception as e:
        print("!!! BACKEND ERROR !!! :", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)