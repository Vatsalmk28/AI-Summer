# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from newspaper import Article
# from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Tokenizer
# import torch
# import io
# import os
# import whisper
# import torchaudio
# from PyPDF2 import PdfReader

# app = Flask(__name__)
# CORS(app)

# # Summarizer setup
# device = 0 if torch.cuda.is_available() else -1
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# # Wav2Vec2 setup
# wav2vec_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# # ---------- Utility functions ----------

# def split_text(text, max_chunk_size=3500):
#     chunks = []
#     while len(text) > max_chunk_size:
#         split_point = text[:max_chunk_size].rfind(".")
#         if split_point == -1:
#             split_point = max_chunk_size
#         chunks.append(text[:split_point + 1])
#         text = text[split_point + 1:]
#     if text:
#         chunks.append(text)
#     return chunks

# def summarize_text(text):
#     chunks = split_text(text)
#     summaries = [summarizer(chunk, max_length=230, min_length=40, do_sample=False)[0]['summary_text'] for chunk in chunks]
#     return " ".join(summaries)

# def transcribe_whisper(file_path, model_size="base"):
#     model = whisper.load_model(model_size)
#     result = model.transcribe(file_path)
#     return result["text"]

# def transcribe_wav2vec2(file_path):
#     waveform, sample_rate = torchaudio.load(file_path)
#     if sample_rate != 16000:
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#         waveform = resampler(waveform)

#     input_values = wav2vec_tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values
#     with torch.no_grad():
#         logits = wav2vec_model(input_values).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     return wav2vec_tokenizer.decode(predicted_ids[0]).lower()

# # ---------- Summarize route (text/pdf/url) ----------

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     try:
#         if 'file' in request.files:
#             file = request.files['file']
#             if file.filename.endswith('.pdf'):
#                 pdf = PdfReader(io.BytesIO(file.read()))
#                 text = ""
#                 for page in pdf.pages:
#                     page_text = page.extract_text()
#                     if page_text:
#                         text += page_text
#                 if not text.strip():
#                     return jsonify({"error": "PDF has no text content"}), 400
#                 summary = summarize_text(text)
#                 return jsonify({
#                     "title": "Uploaded PDF",
#                     "authors": ["Unknown"],
#                     "publish_date": "Unknown",
#                     "summary": summary
#                 })
#             else:
#                 return jsonify({"error": "Unsupported file type in /summarize"}), 400

#         data = request.get_json()

#         if 'url' in data:
#             url = data['url']
#             article = Article(url)
#             article.download()
#             article.parse()
#             title = article.title
#             authors = article.authors
#             publish_date = str(article.publish_date) if article.publish_date else "Unknown"
#             text = article.text

#             if not text.strip():
#                 return jsonify({"error": "Article has no content"}), 400

#             summary = summarize_text(text)
#             return jsonify({
#                 "title": title,
#                 "authors": authors if authors else ["Unknown"],
#                 "publish_date": publish_date,
#                 "summary": summary
#             })

#         elif 'text' in data:
#             text = data['text']
#             if not text.strip():
#                 return jsonify({"error": "Empty text input"}), 400
#             summary = summarize_text(text)
#             return jsonify({
#                 "title": "Raw Text Input",
#                 "authors": ["User"],
#                 "publish_date": "N/A",
#                 "summary": summary
#             })

#         else:
#             return jsonify({"error": "No valid input provided"}), 400

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ---------- Speech Transcription Route ----------

# @app.route('/speech', methods=['POST'])
# def speech():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No audio file uploaded"}), 400

#         file = request.files['file']
#         filename = file.filename.lower()
#         if not filename.endswith((".mp3", ".wav", ".m4a")):
#             return jsonify({"error": "Unsupported audio format"}), 400

#         file_path = os.path.join("temp_" + filename)
#         file.save(file_path)

#         model_choice = request.form.get("model", "whisper").lower()

#         if model_choice == "wav2vec2":
#             transcription = transcribe_wav2vec2(file_path)
#         else:
#             transcription = transcribe_whisper(file_path, model_size="base")

#         os.remove(file_path)
#         return jsonify({
#             "transcription": transcription,
#             "model_used": model_choice
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ---------- Main ----------

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from newspaper import Article
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Tokenizer,GPT2LMHeadModel, GPT2Tokenizer
import torch
import io
import os
import whisper
import torchaudio
from PyPDF2 import PdfReader
from PIL import Image
import numpy as np
import tensorflow as tf
import tempfile

app = Flask(__name__)
CORS(app)

# ---------- Summarizer setup ----------
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# ---------- Wav2Vec2 setup ----------
wav2vec_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# ---------- NST helper functions ----------
def load_uploaded_image(file):
    img = Image.open(file).convert("RGB")
    img = np.array(img) / 255.0
    img = tf.image.resize(img, (512, 512))
    img = img[tf.newaxis, :]
    return tf.convert_to_tensor(img, dtype=tf.float32)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
        self.vgg = tf.keras.Model([vgg.input], outputs)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_features = [gram_matrix(o) for o in style_outputs]
        return {'content': content_outputs, 'style': style_features}

# ---------- Summarizer Utilities ----------
def split_text(text, max_chunk_size=3500):
    chunks = []
    while len(text) > max_chunk_size:
        split_point = text[:max_chunk_size].rfind(".")
        if split_point == -1:
            split_point = max_chunk_size
        chunks.append(text[:split_point + 1])
        text = text[split_point + 1:]
    if text:
        chunks.append(text)
    return chunks

def summarize_text(text):
    chunks = split_text(text)
    summaries = [summarizer(chunk, max_length=230, min_length=40, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

def transcribe_whisper(file_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(file_path)
    return result["text"]

def transcribe_wav2vec2(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    input_values = wav2vec_tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values
    with torch.no_grad():
        logits = wav2vec_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return wav2vec_tokenizer.decode(predicted_ids[0]).lower()

# Load VGG model & define layers just once
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def load_img(file_stream):
    max_dim = 512
    img = tf.io.decode_image(file_stream.read(), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def vgg_layers(layer_names):
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    model.trainable = False
    return model

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_features = [gram_matrix(style_output) for style_output in style_outputs]
        content_features = [content_output for content_output in content_outputs]
        return {'content': content_features, 'style': style_features}

style_extractor = StyleContentModel(style_layers, content_layers)
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image, style_targets, content_targets):
    with tf.GradientTape() as tape:
        outputs = style_extractor(image)
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[i] - style_targets[i])**2)
                               for i in range(num_style_layers)])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[i] - content_targets[i])**2)
                                 for i in range(num_content_layers)])
        content_loss *= content_weight / num_content_layers

        tv_loss = tf.image.total_variation(image)

        total_loss = style_loss + content_loss + total_variation_weight * tv_loss

    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))
    
# Load GPT-2 model & tokenizer once at startup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def generate_text_gpt2(topic, max_length=200):
    input_ids = tokenizer.encode(topic, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------- Routes ----------

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.pdf'):
                pdf = PdfReader(io.BytesIO(file.read()))
                text = "".join(page.extract_text() or "" for page in pdf.pages)
                if not text.strip():
                    return jsonify({"error": "PDF has no text content"}), 400
                summary = summarize_text(text)
                return jsonify({"title": "Uploaded PDF", "authors": ["Unknown"], "publish_date": "Unknown", "summary": summary})
            else:
                return jsonify({"error": "Unsupported file type in /summarize"}), 400

        data = request.get_json()
        if 'url' in data:
            url = data['url']
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            if not text.strip():
                return jsonify({"error": "Article has no content"}), 400
            summary = summarize_text(text)
            return jsonify({"title": article.title, "authors": article.authors or ["Unknown"], "publish_date": str(article.publish_date) or "Unknown", "summary": summary})

        elif 'text' in data:
            text = data['text']
            if not text.strip():
                return jsonify({"error": "Empty text input"}), 400
            summary = summarize_text(text)
            return jsonify({"title": "Raw Text Input", "authors": ["User"], "publish_date": "N/A", "summary": summary})

        else:
            return jsonify({"error": "No valid input provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speech', methods=['POST'])
def speech():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        file = request.files['file']
        file_path = os.path.join("temp_" + file.filename)
        file.save(file_path)

        model_choice = request.form.get("model", "whisper").lower()
        if model_choice == "wav2vec2":
            transcription = transcribe_wav2vec2(file_path)
        else:
            transcription = transcribe_whisper(file_path)

        os.remove(file_path)
        return jsonify({"transcription": transcription, "model_used": model_choice})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/artstyle", methods=["POST"])
def artstyle():
    if "content_image" not in request.files or "style_image" not in request.files:
        return jsonify({"error": "Both content_image and style_image are required."}), 400

    content_image = load_img(request.files["content_image"])
    style_image = load_img(request.files["style_image"])

    style_targets = style_extractor(style_image)['style']
    content_targets = style_extractor(content_image)['content']

    generated_image = tf.Variable(content_image)

    epochs = 3
    steps_per_epoch = 50

    for i in range(epochs):
        for j in range(steps_per_epoch):
            train_step(generated_image, style_targets, content_targets)
            print(f"Epoch {i+1}/{epochs}, Step {j+1}/{steps_per_epoch}")

    # Convert to PIL Image
    final_image_array = (generated_image.numpy().squeeze() * 255).astype(np.uint8)
    pil_image = Image.fromarray(final_image_array)

    # Save to BytesIO
    img_io = io.BytesIO()
    pil_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

def generate_text_gpt2(topic, max_length=200):
    input_ids = tokenizer.encode(topic, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route("/textgen", methods=["POST"])
def textgen():
    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Please provide a 'topic' in the JSON body"}), 400

    topic = data["topic"]
    max_length = data.get("max_length", 200)

    try:
        generated_text = generate_text_gpt2(topic, max_length=max_length)
        return jsonify({"topic": topic, "generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

