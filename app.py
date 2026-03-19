import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, MT5ForSequenceClassification

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
LABEL_NAMES = [
    "Positive",
    "Negative",
    "Mixed_feelings",
    "unknown_state",
    "not-Tamil",
]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"MODEL_PATH is not a directory: {MODEL_PATH}. "
            "Mount or copy your mt5_finetuned checkpoint here."
        )
    print(f"Loading tokenizer & model from {MODEL_PATH} (device={_device}) ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    _model = MT5ForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.to(_device)
    _model.eval()
    print("Model ready.")


def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return jsonify(
            {
                "service": "Tamil–English code-switched sentiment (mT5 full fine-tune)",
                "endpoints": {
                    "GET /health": "liveness check",
                    "POST /predict": 'JSON body: {"text": "your comment here"}',
                },
            }
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            _load()
        except Exception as e:
            return jsonify({"error": str(e)}), 503

        data = request.get_json(silent=True) or {}
        text = data.get("text")
        if text is None or not str(text).strip():
            return jsonify({"error": "Provide non-empty JSON field 'text'."}), 400

        text = str(text).strip()
        enc = _tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(_device) for k, v in enc.items()}

        with torch.no_grad():
            out = _model(**enc)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_id = int(logits.argmax(dim=-1).item())

        return jsonify(
            {
                "label_id": pred_id,
                "label_name": LABEL_NAMES[pred_id],
                "probabilities": {
                    LABEL_NAMES[i]: round(float(probs[i]), 4)
                    for i in range(len(LABEL_NAMES))
                },
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    _load()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
