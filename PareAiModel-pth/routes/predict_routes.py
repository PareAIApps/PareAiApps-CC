from flask import Blueprint, request, jsonify
from datetime import datetime
import torch
import torch.nn.functional as F
import logging

from model.Models_runer import get_model, preprocess_image
from utils.disease import load_disease_data, update_disease_field, load_artikel_data, update_artikel_field
from config import CONFIDENCE_THRESHOLD

predict_bp = Blueprint('predict_bp', __name__)
model = get_model()

@predict_bp.route('/predict', methods=['GET'])
def health_check_predict():
    return jsonify({'message': 'API Active'}), 200

@predict_bp.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model gagal dimuat.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar dalam request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'File tidak dipilih'}), 400

    try:
        img_tensor = preprocess_image(file)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs[0], dim=0)
            class_index = torch.argmax(probs).item()
            confidence = probs[class_index].item()

        created_at = datetime.now().isoformat()

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'class_label': "Objek Tidak Tersedia.",
                'confidence': confidence,
                'suggestion': "Pastikan gambar jelas dan fokus pada objek yang ingin diprediksi.",
                'description': "Model tidak yakin dengan hasil prediksi. Mungkin gambar buram atau objek tidak relevan.",
                'tools_receipt': "-",
                'tutorial': "-",
                'createdAt': created_at
            })

        disease_data = load_disease_data()
        disease = disease_data.get(str(class_index), {})

        return jsonify({
            'class_label': disease.get("label", "Unknown"),
            'confidence': confidence,
            'suggestion': disease.get("suggestion", "Tidak ada saran tersedia."),
            'description': disease.get("description", "Tidak ada deskripsi tersedia."),
            'tools_receipt': disease.get("tools_materials", "-"),
            'tutorial': disease.get("tutorial_steps", "-"),
            'createdAt': created_at
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'Terjadi kesalahan saat memproses gambar'}), 500


# === UPDATE ENDPOINTS ===
@predict_bp.route('/disease/<string:label>/description', methods=['PUT'])
def update_description(label):
    data = request.json
    if update_disease_field(label, 'description', data.get('description')):
        return jsonify({'message': 'Description updated successfully.'})
    return jsonify({'error': 'Label not found.'}), 404

@predict_bp.route('/disease/<string:label>/suggestion', methods=['PUT'])
def update_suggestion(label):
    data = request.json
    if update_disease_field(label, 'suggestion', data.get('suggestion')):
        return jsonify({'message': 'Suggestion updated successfully.'})
    return jsonify({'error': 'Label not found.'}), 404

@predict_bp.route('/disease/<string:label>/tools_materials', methods=['PUT'])
def update_tools_materials(label):
    data = request.json
    new_tools = data.get('tools_materials')
    if not isinstance(new_tools, str):
        return jsonify({'error': 'tools_materials harus berupa string (dengan newline \\n jika diperlukan)'}), 400
    if update_disease_field(label, 'tools_materials', new_tools):
        return jsonify({'message': 'Tools/materials updated successfully.'})
    return jsonify({'error': 'Label not found.'}), 404

@predict_bp.route('/disease/<string:label>/tutorial_steps', methods=['PUT'])
def update_tutorial_steps(label):
    data = request.json
    new_steps = data.get('tutorial_steps')
    if not isinstance(new_steps, str):
        return jsonify({'error': 'tutorial_steps harus berupa string (dengan newline \\n jika diperlukan)'}), 400
    if update_disease_field(label, 'tutorial_steps', new_steps):
        return jsonify({'message': 'Tutorial steps updated successfully.'})
    return jsonify({'error': 'Label not found.'}), 404

# === ARTIKELS ENDPOINTS ===

@predict_bp.route('/artikels', methods=['GET'])
def get_all_artikels():
    data = load_artikel_data()
    return jsonify(data), 200

@predict_bp.route('/artikels/<string:label>/description', methods=['PUT'])
def update_artikel_description(label):
    data = request.json
    description = data.get("description")
    if not description:
        return jsonify({"error": "Field 'description' is required"}), 400
    updates = {
        "description": description,
        "created_at": datetime.now().strftime("%d %B %Y")
    }
    if update_artikel_field(label, updates):
        return jsonify({"message": f"Artikel '{label}' description updated successfully."}), 200
    return jsonify({"error": f"Artikel '{label}' not found."}), 404

@predict_bp.route('/artikels/<string:label>/subtitle', methods=['PUT'])
def update_artikel_subtitle(label):
    data = request.json
    subtitle = data.get("subtitle")
    if not subtitle:
        return jsonify({"error": "Field 'subtitle' is required"}), 400
    updates = {
        "subtitle": subtitle,
        "created_at": datetime.now().strftime("%d %B %Y")
    }
    if update_artikel_field(label, updates):
        return jsonify({"message": f"Artikel '{label}' subtitle updated successfully."}), 200
    return jsonify({"error": f"Artikel '{label}' not found."}), 404

@predict_bp.route('/artikels/<string:label>/image_url', methods=['PUT'])
def update_artikel_image_url(label):
    data = request.json
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "Field 'image_url' is required"}), 400
    updates = {
        "image_url": image_url,
        "created_at": datetime.now().strftime("%d %B %Y")
    }
    if update_artikel_field(label, updates):
        return jsonify({"message": f"Artikel '{label}' image_url updated successfully."}), 200
    return jsonify({"error": f"Artikel '{label}' not found."}), 404