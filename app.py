import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from pathlib import Path
from scipy import stats
import base64 
import threading
import os
from dotenv import load_dotenv

# --- SMART IMPORT BLOCK (CRITICAL FOR RENDER) ---
try:
    # 1. Try Lightweight Runtime (For Render/Linux)
    import tflite_runtime.interpreter as tflite
    print("--- Using TFLite Runtime (Lightweight) ---")
except ImportError:
    # 2. Fallback to Full TensorFlow (For Local Dev)
    try:
        import tensorflow.lite as tflite
        print("--- Using Full TensorFlow Lite (Local Fallback) ---")
    except ImportError:
        print("CRITICAL ERROR: 'tflite_runtime' not found. Ensure it is in requirements.txt")

load_dotenv()

# --- AUTH & DB IMPORTS ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Scan 

# --- EMAIL IMPORT (Uses your new Brevo logic) ---
try:
    from email_sending import send_report_email
except ImportError:
    print("Warning: email_sending.py not found.")

# --- CUSTOM MODULE IMPORTS ---
try:
    from chessboard_snipper import process_image
    from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen
except ImportError:
    print("CRITICAL: Missing helper modules")

app = Flask(__name__)

# --- CONFIGURATION ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')

# Database Config
database_url = os.environ.get('DATABASE_URL', 'sqlite:///chessvision.db')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Fine_tuned_CNN_Model" / "chess_model_v5.tflite"
LABELS_PATH = BASE_DIR / "labels" / "class_names.txt"

# --- INIT EXTENSIONS ---
db.init_app(app) 
login_manager = LoginManager()
login_manager.login_view = 'login' 
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# --- TFLITE RESOURCES ---
INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None
CLASS_NAMES = None

def load_resources():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, CLASS_NAMES
    
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        print(f"--- Loading TFLite Model: {MODEL_PATH} ---")
        try:
            INTERPRETER = tflite.Interpreter(model_path=str(MODEL_PATH))
            INTERPRETER.allocate_tensors()
            
            INPUT_DETAILS = INTERPRETER.get_input_details()
            OUTPUT_DETAILS = INTERPRETER.get_output_details()
            
            CLASS_NAMES = LABELS_PATH.read_text().splitlines()
            print("--- System Ready (TFLite Mode) ---")
        except Exception as e:
            print(f"CRITICAL MODEL ERROR: {e}")
    else:
        print(f"ERROR: Model/Labels not found at {MODEL_PATH}")

# --- HELPER: TFLite Inference ---
def tflite_predict(interpreter, input_data):
    input_index = INPUT_DETAILS[0]['index']
    output_index = OUTPUT_DETAILS[0]['index']
    predictions = []
    
    for i in range(len(input_data)):
        img = input_data[i:i+1].astype(np.float32) 
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        predictions.append(output[0])
        
    return np.array(predictions)

def predict_with_voting(interpreter, squares_batch):
    augmented_squares = []
    for sq in squares_batch:
        augmented_squares.append(sq) 
        augmented_squares.append(np.roll(sq, -2, axis=1))
        augmented_squares.append(np.roll(sq, -2, axis=0))
        augmented_squares.append(np.clip(sq * 0.7, 0, 255))
        
        crop = sq[4:60, 4:60]
        aug_zoom = cv2.resize(crop, (64, 64))
        augmented_squares.append(aug_zoom)
    
    big_batch = np.array(augmented_squares)
    
    # Use TFLite helper
    preds = tflite_predict(interpreter, big_batch)
    
    num_classes = preds.shape[1]
    reshaped_preds = preds.reshape(64, 5, num_classes)
    
    final_indices = []
    for i in range(64):
        votes = np.argmax(reshaped_preds[i], axis=1)
        winner = stats.mode(votes, keepdims=True).mode[0]
        final_indices.append(winner)
    return final_indices

def correct_color_errors(image_rgb, predicted_label):
    if "empty" in predicted_label: return predicted_label
    parts = predicted_label.split('_')
    if len(parts) < 2: return predicted_label
    
    current_color = parts[0]
    piece_type = parts[1]
    
    if image_rgb.dtype != np.uint8: img_uint8 = image_rgb.astype(np.uint8)
    else: img_uint8 = image_rgb
    
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = gray[h//4 : 3*h//4, w//4 : 3*w//4]
    brightness = np.mean(center)
    
    if brightness < 60 and current_color == "light": return f"dark_{piece_type}"
    if brightness > 180 and current_color == "dark": return f"light_{piece_type}"
    return predicted_label

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', user=current_user)

@app.route('/predict', methods=['POST'])
def predict():
    if INTERPRETER is None: return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    pov = request.form.get('pov', 'w')

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return jsonify({'error': 'Invalid image file'}), 400

        processed = process_image(img)
        if processed is None: return jsonify({'error': 'No chessboard detected.'}), 400
        
        model_inputs, board_viz, _ = processed
        
        # INFERENCE (TFLite)
        pred_indices = predict_with_voting(INTERPRETER, model_inputs)
        pred_labels = [CLASS_NAMES[i] for i in pred_indices]

        final_labels = []
        for i, label in enumerate(pred_labels):
            corrected = correct_color_errors(model_inputs[i], label)
            final_labels.append(corrected)

        fen = assemble_fen_from_predictions(final_labels)
        
        # Server-side Logic (Keep this simple, client handles display flip)
        turn = 'w'
        if pov == 'b':
            fen = black_perspective_fen(fen)
            turn = 'b'
            
        final_fen = f"{fen} {turn} KQkq - 0 1"

        is_success, buffer = cv2.imencode(".jpg", board_viz)
        cropped_image_data = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}" if is_success else None

        # DB Save
        if current_user.is_authenticated:
            try:
                new_scan = Scan(fen=final_fen, image_data=cropped_image_data or "", user_id=current_user.id)
                db.session.add(new_scan)
                db.session.commit()
            except Exception as db_e:
                print(f"DB Error: {db_e}")

        return jsonify({'fen': final_fen, 'cropped_image': cropped_image_data})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

# --- UPDATED REPORT ISSUE ROUTE ---
@app.route('/report_issue', methods=['POST'])
def report_issue():
    # 1. Text Data
    tags = request.form.get('tags', 'General')
    text = request.form.get('feedback', '')
    fen = request.form.get('fen', 'N/A')
    
    # 2. Files
    orig = request.files.get('original_image')
    crop = request.files.get('cropped_image')
    bug_file = request.files.get('attachment') # New attachment

    # 3. Read Bytes
    orig_bytes = orig.read() if orig else None
    crop_bytes = crop.read() if crop else None
    attach_bytes = bug_file.read() if bug_file else None

    # 4. Threading (Pass all 6 args to email_sending.py)
    threading.Thread(
        target=send_report_email,
        args=(text, tags, fen, orig_bytes, crop_bytes, attach_bytes)
    ).start()

    return jsonify({"status": "success"})

# --- AUTH & HISTORY ROUTES ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
        elif User.query.filter_by(username=username).first():
            flash('Username taken.', 'error')
        else:
            new_user = User(email=email, username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        flash('Invalid credentials.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/history')
@login_required
def get_history():
    try:
        scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(5).all()
        return jsonify([{'fen': s.fen, 'image': s.image_data, 'date': s.timestamp.strftime("%b %d, %H:%M")} for s in scans])
    except:
        return jsonify([])

# --- INIT ---
with app.app_context():
    db.create_all()

load_resources()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
