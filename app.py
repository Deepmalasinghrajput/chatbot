from flask import Flask, render_template, jsonify, request
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

app = Flask(__name__)

# load model and tokenizer
model_path = "artifacts/Chatbot_model"

model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

device = model.device


# clean the text
def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = text.strip().lower()
    return text


# Chatbot function
def chatbot(dialogue):

    dialogue = clean_text(dialogue)

    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=250
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'],
        max_length=250,
        num_beams=4,
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json.get("message", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    response = chatbot(user_message)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)