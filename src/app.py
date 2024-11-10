import os
from flask import Flask, request, jsonify

from src.classifier import classify_document, sbert_class_embeddings, clip_class_embeddings
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify_file', methods=['POST'])
def classify_file_route():

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    file_path = os.path.join("/tmp", file.filename)
    file.save(file_path)

    file_type = file.filename.rsplit('.', 1)[1].lower()

    file_class, confidence, scores = classify_document(file_path, file_type, sbert_class_embeddings, clip_class_embeddings)

    os.remove(file_path)

    return jsonify({"file_class": file_class, "confidence": str(confidence)}), 200

if __name__ == '__main__':
    app.run(debug=True)
