from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from predictor import MultiWriterPredictor 
from flask_cors import CORS
import cv2

UPLOAD_FOLDER = "uploads"
MODEL_NAME_PREFIX = "multi_writer_detector_final_20250724_102957.h5"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, supports_credentials=True)

model_files = [f for f in os.listdir('.') if f.startswith(MODEL_NAME_PREFIX) and f.endswith('.h5')]
if not model_files:
    raise FileNotFoundError("Modelul .h5 nu a fost găsit.")
model_path = max(model_files, key=os.path.getctime)
predictor = MultiWriterPredictor(model_path)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return "Serverul este activ."


@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nicio imagine trimisă'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Numele fișierului este gol'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)
        gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(filepath, gray_img)
        # Citește parametru de mod (full sau patches)
        mode = request.args.get('mode', 'full').lower().strip()

        try:
            if mode == 'patches':
                patch_results = predictor.predict_by_two_word_patches(filepath, show_all=False)
                if patch_results is None or not patch_results:
                    return jsonify({'error': 'Nu s-au putut extrage patch-uri valide'}), 500
                
                json_results = [{
                    'patch_index': r['patch_index'],
                    'result': r['result_text'],
                    'confidence': float(round(r['confidence'], 2)),
                    'raw_score': float(round(r['prediction_proba'], 4)),
                    'class': int(r['prediction_class'])
                } for r in patch_results]

                return jsonify({
                    'filename': filename,
                    'mode': 'patches',
                    'num_patches': len(json_results),
                    'patches': json_results
                })

            else:  # default: full image
                result = predictor.predict_single_image(filepath, show_image=False)
                return jsonify({
                    'filename': filename,
                    'mode': 'full',
                    'result': result['result_text'],
                    'confidence': float(round(result['confidence'], 2)),
                    'raw_score': float(round(result['prediction_proba'], 4)),
                    'class': int(result['prediction_class'])
                })

        except Exception as e:
            print(f"EROARE LA PREDICTIE: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Fișierul nu este o imagine validă'}), 400


if __name__ == "__main__":
    app.run(debug=True)
