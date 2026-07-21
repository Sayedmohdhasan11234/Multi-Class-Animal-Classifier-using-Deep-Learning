import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model (TFLite — lightweight, no TensorFlow required at runtime)
# ---------------------------------------------------------------------------
MODEL_PATH = "animal_classifier.tflite"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' not found. Run convert_to_tflite.py (with full "
        f"TensorFlow available) to produce it, then place it in this folder."
    )

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    "antelope", "bat", "beaver", "blue+whale", "bobcat", "buffalo",
    "chihuahua", "cow", "dalmatian", "deer", "dolphin", "elephant",
    "german+shepherd", "giant+panda", "giraffe", "grizzly+bear", "hamster",
    "hippopotamus", "humpback+whale", "killer+whale", "leopard", "lion",
    "mole", "mouse", "otter", "ox", "persian+cat", "pig", "polar+bear",
    "raccoon", "rat", "seal", "siamese+cat", "skunk", "spider+monkey",
    "tiger", "walrus", "weasel", "wolf", "zebra"
]

IMG_SIZE = (224, 224)


def format_label(name):
    """'blue+whale' -> 'Blue Whale'"""
    return name.replace("+", " ").title()


def prepare_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize(IMG_SIZE)
    # Note: tf.keras.applications.efficientnet.preprocess_input is a no-op
    # (EfficientNet's rescaling/normalization is baked into the model
    # itself), so raw 0-255 float pixels are the correct input here too.
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_inference(img_array):
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_array = prepare_image(file)
        prediction = run_inference(img_array)

        top_idx = int(np.argmax(prediction))
        top3_idx = np.argsort(prediction)[::-1][:3]

        top3 = [
            {
                "label": format_label(CLASS_NAMES[i]),
                "confidence": round(float(prediction[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        return jsonify({
            "animal": format_label(CLASS_NAMES[top_idx]),
            "confidence": round(float(prediction[top_idx]) * 100, 2),
            "top3": top3
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
