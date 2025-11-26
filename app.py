# import tensorflow as tf
# import numpy as np
# import cv2
# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
# from pathlib import Path
# from scipy import stats
# import base64 
# import smtplib
# import threading
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.image import MIMEImage
# import os

# from dotenv import load_dotenv
# load_dotenv()

# # --- NEW: AUTH & DB IMPORTS ---
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, login_user, login_required, logout_user, current_user
# from werkzeug.security import generate_password_hash, check_password_hash
# # Import models from the file you created
# from models import db, User, Scan 

# # --- CUSTOM MODULE IMPORTS ---
# try:
#     from chessboard_snipper import process_image
#     from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen
# except ImportError:
#     print("CRITICAL: Missing 'chessboard_snipper.py' or 'flip_board_to_black_pov.py'")

# app = Flask(__name__)

# # --- CONFIGURATION ---
# # Security Key
# app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# # Database: Checks for Render's DATABASE_URL, falls back to local SQLite
# database_url = os.environ.get('DATABASE_URL', 'sqlite:///chessvision.db')
# if database_url and database_url.startswith("postgres://"):
#     database_url = database_url.replace("postgres://", "postgresql://", 1)

# app.config['SQLALCHEMY_DATABASE_URI'] = database_url
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # --- PATH CONFIGURATION (FIXED) ---
# # We use pathlib to find the folder where app.py is located
# BASE_DIR = Path(__file__).resolve().parent

# # Now we construct the paths relative to that folder. 
# # This works on Windows, Mac, Linux, and Render.
# MODEL_PATH = BASE_DIR / "Fine_tuned_CNN_Model" / "chess_model_v4.keras"
# LABELS_PATH = BASE_DIR / "labels" / "class_names.txt"

# # Email Config
# EMAIL_SENDER = os.environ.get('EMAIL_SENDER')
# EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
# EMAIL_RECEIVER = os.environ.get('EMAIL_RECEIVER')

# # --- INIT EXTENSIONS ---
# db.init_app(app) 

# login_manager = LoginManager()
# login_manager.login_view = 'login' 
# login_manager.init_app(app)

# @login_manager.user_loader
# def load_user(id):
#     return User.query.get(int(id))

# # --- MODEL LOADING & PATCHING ---
# MODEL = None
# CLASS_NAMES = None

# class PatchedRandomContrast(tf.keras.layers.RandomContrast):
#     def __init__(self, factor, value_range=None, **kwargs):
#         super().__init__(factor, **kwargs)

# def load_resources():
#     global MODEL, CLASS_NAMES
#     # Path objects allow .exists() check
#     if MODEL_PATH.exists() and LABELS_PATH.exists():
#         print(f"--- Loading Model from: {MODEL_PATH} ---")
#         try:
#             MODEL = tf.keras.models.load_model(MODEL_PATH)
#         except (ValueError, TypeError) as e:
#             print(f"Applying patch... {e}")
#             with tf.keras.utils.custom_object_scope({'RandomContrast': PatchedRandomContrast}):
#                 MODEL = tf.keras.models.load_model(MODEL_PATH)
        
#         # Path objects allow .read_text()
#         CLASS_NAMES = LABELS_PATH.read_text().splitlines()
#         print("--- System Ready ---")
#     else:
#         print(f"ERROR: Could not find model/labels.")
#         print(f"Looked for: {MODEL_PATH}")
#         print(f"Looked for: {LABELS_PATH}")

# # --- AUTH ROUTES ---

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         username = request.form.get('username')
#         password = request.form.get('password')

#         email_exists = User.query.filter_by(email=email).first()
#         username_exists = User.query.filter_by(username=username).first()

#         if email_exists:
#             flash('Email already registered. Please log in.', 'error')
#         elif username_exists:
#             flash('Username is already taken. Please choose another.', 'error')
#         else:
#             hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
#             new_user = User(email=email, username=username, password=hashed_pw)
            
#             try:
#                 db.session.add(new_user)
#                 db.session.commit()
#                 login_user(new_user) 
#                 return redirect(url_for('index'))
#             except Exception as e:
#                 flash(f'Error creating account: {e}', 'error')
            
#     return render_template('signup.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         user = User.query.filter_by(email=email).first()

#         if user and check_password_hash(user.password, password):
#             login_user(user, remember=True)
#             return redirect(url_for('index'))
#         else:
#             flash('Incorrect email or password.', 'error')

#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('index'))

# # --- API: HISTORY ---
# @app.route('/api/history')
# @login_required
# def get_history():
#     try:
#         user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(5).all()
        
#         history_data = []
#         for scan in user_scans:
#             history_data.append({
#                 'fen': scan.fen,
#                 'image': scan.image_data, 
#                 'date': scan.timestamp.strftime("%b %d, %H:%M")
#             })
#         return jsonify(history_data)
#     except Exception as e:
#         print(f"History Error: {e}")
#         return jsonify([])

# # --- MAIN APP LOGIC ---

# @app.route('/')
# def index():
#     return render_template('index.html', user=current_user)

# def predict_with_voting(model, squares_batch):
#     augmented_squares = []
#     for sq in squares_batch:
#         augmented_squares.append(sq) 
#         augmented_squares.append(np.roll(sq, -2, axis=1))
#         augmented_squares.append(np.roll(sq, -2, axis=0))
#         augmented_squares.append(np.clip(sq * 0.7, 0, 255))
#         h, w = 64, 64
#         crop = sq[4:60, 4:60]
#         aug_zoom = cv2.resize(crop, (64, 64))
#         augmented_squares.append(aug_zoom)
    
#     big_batch = np.array(augmented_squares)
#     preds = model.predict(big_batch, verbose=0)
#     num_classes = preds.shape[1]
#     reshaped_preds = preds.reshape(64, 5, num_classes)
    
#     final_indices = []
#     for i in range(64):
#         votes = np.argmax(reshaped_preds[i], axis=1)
#         winner = stats.mode(votes, keepdims=True).mode[0]
#         final_indices.append(winner)
#     return final_indices

# def correct_color_errors(image_rgb, predicted_label):
#     if "empty" in predicted_label: return predicted_label
#     piece_type = predicted_label.split('_')[1]
#     current_color = predicted_label.split('_')[0]
    
#     if image_rgb.dtype != np.uint8: img_uint8 = image_rgb.astype(np.uint8)
#     else: img_uint8 = image_rgb
    
#     img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape
#     center = gray[h//4 : 3*h//4, w//4 : 3*w//4]
#     brightness = np.mean(center)
    
#     if brightness < 60 and current_color == "light": return f"dark_{piece_type}"
#     if brightness > 180 and current_color == "dark": return f"light_{piece_type}"
#     return predicted_label

# @app.route('/predict', methods=['POST'])
# def predict():
#     if MODEL is None: return jsonify({'error': 'Model not loaded'}), 500
#     if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     pov = request.form.get('pov', 'w')

#     try:
#         img_bytes = file.read()
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img is None: return jsonify({'error': 'Invalid image file'}), 400

#         processed = process_image(img)
#         if processed is None: return jsonify({'error': 'No chessboard detected.'}), 400
        
#         model_inputs, board_viz, _ = processed
        
#         pred_indices = predict_with_voting(MODEL, model_inputs)
#         pred_labels = [CLASS_NAMES[i] for i in pred_indices]

#         final_labels = []
#         for i, label in enumerate(pred_labels):
#             corrected = correct_color_errors(model_inputs[i], label)
#             final_labels.append(corrected)

#         fen = assemble_fen_from_predictions(final_labels)
#         if pov == 'b':
#             fen = black_perspective_fen(fen)
#             turn = 'b'
#         else:
#             turn = 'w'
#         final_fen = f"{fen} {turn} KQkq - 0 1"

#         is_success, buffer = cv2.imencode(".jpg", board_viz)
#         if is_success:
#             base64_image = base64.b64encode(buffer).decode('utf-8')
#             cropped_image_data = f"data:image/jpeg;base64,{base64_image}"
#         else:
#             cropped_image_data = None

#         if current_user.is_authenticated:
#             try:
#                 new_scan = Scan(
#                     fen=final_fen, 
#                     image_data=cropped_image_data or "", 
#                     user_id=current_user.id
#                 )
#                 db.session.add(new_scan)
#                 db.session.commit()
#             except Exception as db_e:
#                 print(f"Warning: DB Save Failed: {db_e}") 

#         return jsonify({
#             'fen': final_fen,
#             'cropped_image': cropped_image_data
#         })

#     except Exception as e:
#         print(e)
#         return jsonify({'error': str(e)}), 500

# def send_email_async(feedback_text, tags, fen, original_img_bytes, crop_img_bytes):
#     try:
#         msg = MIMEMultipart()
#         msg['From'] = EMAIL_SENDER
#         msg['To'] = EMAIL_RECEIVER
#         msg['Subject'] = f"[SnapFen Report] {tags}"
#         body = f"<h3>Feedback</h3><p>{feedback_text}</p><p>FEN: {fen}</p>"
#         msg.attach(MIMEText(body, 'html'))
        
#         if original_img_bytes:
#             img1 = MIMEImage(original_img_bytes, name="original.png")
#             msg.attach(img1)
#         if crop_img_bytes:
#             img2 = MIMEImage(crop_img_bytes, name="crop.png")
#             msg.attach(img2)

#         with smtplib.SMTP('smtp.gmail.com', 587) as server:
#             server.starttls()
#             server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#             server.send_message(msg)
#     except Exception as e:
#         print(f"Email Error: {e}")

# @app.route('/report_issue', methods=['POST'])
# def report_issue():
#     tags = request.form.get('tags', 'General')
#     feedback = request.form.get('feedback', 'No details')
#     fen = request.form.get('fen', 'N/A')
    
#     orig_file = request.files.get('original_image')
#     crop_file = request.files.get('cropped_image') 
    
#     orig_bytes = orig_file.read() if orig_file else None
#     crop_bytes = crop_file.read() if crop_file else None

#     thread = threading.Thread(target=send_email_async, args=(feedback, tags, fen, orig_bytes, crop_bytes))
#     thread.start()

#     return jsonify({'status': 'success'})

# # --- PRODUCTION INIT ---
# with app.app_context():
#     db.create_all()
#     print("--- Database Tables Checked/Created ---")

# load_resources()

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)



# Adding code compatible with tensorflow lite version


import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from pathlib import Path
import io
from scipy import stats
import base64 
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from dotenv import load_dotenv

# --- NEW IMPORT: LIGHTWEIGHT TFLITE ---
# We use tflite_runtime instead of the massive tensorflow library
import tflite_runtime.interpreter as tflite

# Load environment variables locally
load_dotenv()

# --- NEW: AUTH & DB IMPORTS ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# Import models from the file you created
from models import db, User, Scan 

# --- CUSTOM MODULE IMPORTS ---
try:
    from chessboard_snipper import process_image
    from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen
except ImportError:
    print("CRITICAL: Missing 'chessboard_snipper.py' or 'flip_board_to_black_pov.py'")

app = Flask(__name__)

# --- CONFIGURATION ---
# Security Key
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chess-vision-secret-key-mvp-2025')

# Database: Checks for Render's DATABASE_URL, falls back to local SQLite
database_url = os.environ.get('DATABASE_URL', 'sqlite:///chessvision.db')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent

# Point to the TFLite model
MODEL_PATH = BASE_DIR / "Fine_tuned_CNN_Model" / "chess_model_v4.tflite"
LABELS_PATH = BASE_DIR / "labels" / "class_names.txt"

# Email Config
EMAIL_SENDER = os.environ.get('EMAIL_SENDER')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
EMAIL_RECEIVER = os.environ.get('EMAIL_RECEIVER')

# --- INIT EXTENSIONS ---
db.init_app(app) 

login_manager = LoginManager()
login_manager.login_view = 'login' 
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# --- TFLITE MODEL LOADING ---
INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None
CLASS_NAMES = None

def load_resources():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, CLASS_NAMES
    
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        print(f"--- Loading TFLite Model from: {MODEL_PATH} ---")
        try:
            # Load TFLite Interpreter (Using the runtime, not full TF)
            INTERPRETER = tflite.Interpreter(model_path=str(MODEL_PATH))
            INTERPRETER.allocate_tensors()
            
            # Get input/output details for inference
            INPUT_DETAILS = INTERPRETER.get_input_details()
            OUTPUT_DETAILS = INTERPRETER.get_output_details()
            
            CLASS_NAMES = LABELS_PATH.read_text().splitlines()
            print("--- System Ready (TFLite Runtime Mode) ---")
        except Exception as e:
            print(f"CRITICAL ERROR loading model: {e}")
    else:
        print(f"ERROR: Could not find model/labels.")
        print(f"Looked for: {MODEL_PATH}")

# --- AUTH ROUTES ---

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        email_exists = User.query.filter_by(email=email).first()
        username_exists = User.query.filter_by(username=username).first()

        if email_exists:
            flash('Email already registered. Please log in.', 'error')
        elif username_exists:
            flash('Username is already taken. Please choose another.', 'error')
        else:
            hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(email=email, username=username, password=hashed_pw)
            
            try:
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user) 
                return redirect(url_for('index'))
            except Exception as e:
                flash(f'Error creating account: {e}', 'error')
            
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
        else:
            flash('Incorrect email or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- API: HISTORY ---
@app.route('/api/history')
@login_required
def get_history():
    try:
        user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(5).all()
        
        history_data = []
        for scan in user_scans:
            history_data.append({
                'fen': scan.fen,
                'image': scan.image_data, 
                'date': scan.timestamp.strftime("%b %d, %H:%M")
            })
        return jsonify(history_data)
    except Exception as e:
        print(f"History Error: {e}")
        return jsonify([])

# --- MAIN APP LOGIC ---

@app.route('/')
def index():
    return render_template('index.html', user=current_user)

# Helper: TFLite Prediction Wrapper
def tflite_predict(interpreter, input_data):
    """
    Runs inference on a batch of images using TFLite.
    input_data shape: (Batch_Size, 64, 64, 3)
    """
    input_index = INPUT_DETAILS[0]['index']
    output_index = OUTPUT_DETAILS[0]['index']
    
    predictions = []
    
    # Loop through each image in the batch
    for i in range(len(input_data)):
        # 1. Prepare single input (1, 64, 64, 3)
        # Ensure it's float32 as expected by the model
        img = input_data[i:i+1].astype(np.float32) 
        
        # 2. Set Tensor
        interpreter.set_tensor(input_index, img)
        
        # 3. Invoke
        interpreter.invoke()
        
        # 4. Get Result
        output = interpreter.get_tensor(output_index)
        predictions.append(output[0])
        
    return np.array(predictions)

def predict_with_voting(interpreter, squares_batch):
    augmented_squares = []
    for sq in squares_batch:
        # Standard TTA augmentations
        augmented_squares.append(sq) 
        augmented_squares.append(np.roll(sq, -2, axis=1))
        augmented_squares.append(np.roll(sq, -2, axis=0))
        augmented_squares.append(np.clip(sq * 0.7, 0, 255))
        
        h, w = 64, 64
        crop = sq[4:60, 4:60]
        aug_zoom = cv2.resize(crop, (64, 64))
        augmented_squares.append(aug_zoom)
    
    big_batch = np.array(augmented_squares)
    
    # USE TFLITE INFERENCE HERE
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
    piece_type = predicted_label.split('_')[1]
    current_color = predicted_label.split('_')[0]
    
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
        
        # Pass INTERPRETER instead of MODEL
        pred_indices = predict_with_voting(INTERPRETER, model_inputs)
        pred_labels = [CLASS_NAMES[i] for i in pred_indices]

        final_labels = []
        for i, label in enumerate(pred_labels):
            corrected = correct_color_errors(model_inputs[i], label)
            final_labels.append(corrected)

        fen = assemble_fen_from_predictions(final_labels)
        if pov == 'b':
            fen = black_perspective_fen(fen)
            turn = 'b'
        else:
            turn = 'w'
        final_fen = f"{fen} {turn} KQkq - 0 1"

        is_success, buffer = cv2.imencode(".jpg", board_viz)
        if is_success:
            base64_image = base64.b64encode(buffer).decode('utf-8')
            cropped_image_data = f"data:image/jpeg;base64,{base64_image}"
        else:
            cropped_image_data = None

        # Save to History if logged in
        if current_user.is_authenticated:
            try:
                new_scan = Scan(
                    fen=final_fen, 
                    image_data=cropped_image_data or "", 
                    user_id=current_user.id
                )
                db.session.add(new_scan)
                db.session.commit()
            except Exception as db_e:
                print(f"Warning: DB Save Failed: {db_e}") 

        return jsonify({
            'fen': final_fen,
            'cropped_image': cropped_image_data
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

# --- EMAIL LOGIC ---
def send_email_async(feedback_text, tags, fen, original_img_bytes, crop_img_bytes):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"[SnapFen Report] {tags}"
        body = f"<h3>Feedback</h3><p>{feedback_text}</p><p>FEN: {fen}</p>"
        msg.attach(MIMEText(body, 'html'))
        
        if original_img_bytes:
            img1 = MIMEImage(original_img_bytes, name="original.png")
            msg.attach(img1)
        if crop_img_bytes:
            img2 = MIMEImage(crop_img_bytes, name="crop.png")
            msg.attach(img2)

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Email Error: {e}")

@app.route('/report_issue', methods=['POST'])
def report_issue():
    tags = request.form.get('tags', 'General')
    feedback = request.form.get('feedback', 'No details')
    fen = request.form.get('fen', 'N/A')
    
    orig_file = request.files.get('original_image')
    crop_file = request.files.get('cropped_image') 
    
    orig_bytes = orig_file.read() if orig_file else None
    crop_bytes = crop_file.read() if crop_file else None

    thread = threading.Thread(target=send_email_async, args=(feedback, tags, fen, orig_bytes, crop_bytes))
    thread.start()

    return jsonify({'status': 'success'})

# --- PRODUCTION INIT ---
with app.app_context():
    db.create_all()
    print("--- Database Tables Checked/Created ---")

load_resources()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
