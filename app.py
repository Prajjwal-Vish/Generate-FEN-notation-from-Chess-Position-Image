# import tensorflow as tf
# import numpy as np
# import cv2
# from flask import Flask, request, render_template, jsonify
# from pathlib import Path
# import io
# from scipy import stats
# import base64 

# # --- CUSTOM MODULE IMPORTS ---
# try:
#     from chessboard_snipper import process_image
#     from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen
# except ImportError:
#     print("CRITICAL: Missing 'chessboard_snipper.py' or 'flip_board_to_black_pov.py'")

# app = Flask(__name__)

# # --- CONFIGURATION ---
# MODEL_PATH = Path(r"C:\Users\prajjwal\Desktop\chess-to-fen\Gererate-FEN-notation-from-Chess-Position-Image\Fine_tuned_CNN_Model\chess_model_v4.keras")
# LABELS_PATH = Path(r"C:\Users\prajjwal\Desktop\chess-to-fen\Gererate-FEN-notation-from-Chess-Position-Image\labels\class_names.txt")

# MODEL = None
# CLASS_NAMES = None

# # --- VERSION COMPATIBILITY FIX ---
# class PatchedRandomContrast(tf.keras.layers.RandomContrast):
#     def __init__(self, factor, value_range=None, **kwargs):
#         super().__init__(factor, **kwargs)

# def load_resources():
#     global MODEL, CLASS_NAMES
#     if MODEL_PATH.exists() and LABELS_PATH.exists():
#         print("--- Loading Model & Labels... ---")
#         try:
#             MODEL = tf.keras.models.load_model(MODEL_PATH)
#         except (ValueError, TypeError) as e:
#             print(f"Applying patch... {e}")
#             with tf.keras.utils.custom_object_scope({'RandomContrast': PatchedRandomContrast}):
#                 MODEL = tf.keras.models.load_model(MODEL_PATH)
        
#         CLASS_NAMES = LABELS_PATH.read_text().splitlines()
#         print("--- System Ready ---")
#     else:
#         print(f"ERROR: Could not find model at {MODEL_PATH}")

# # --- LOGIC ---

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

#     if image_rgb.dtype != np.uint8:
#         img_uint8 = image_rgb.astype(np.uint8)
#     else:
#         img_uint8 = image_rgb

#     img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#     h, w = gray.shape
#     center = gray[h//4 : 3*h//4, w//4 : 3*w//4]
#     brightness = np.mean(center)

#     if brightness < 60 and current_color == "light":
#         return f"dark_{piece_type}"
#     if brightness > 180 and current_color == "dark":
#         return f"light_{piece_type}"

#     return predicted_label

# # --- ROUTES ---

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if MODEL is None: return jsonify({'error': 'Model not loaded'}), 500
#     if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     pov = request.form.get('pov', 'w')

#     try:
#         # 1. Read Image
#         img_bytes = file.read()
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if img is None: return jsonify({'error': 'Invalid image file'}), 400

#         # 2. Snip Board
#         processed = process_image(img)
#         if processed is None: return jsonify({'error': 'No chessboard detected. Try cropping manually.'}), 400
        
#         model_inputs, board_viz, _ = processed

#         # 3. Predict with Voting
#         pred_indices = predict_with_voting(MODEL, model_inputs)
#         pred_labels = [CLASS_NAMES[i] for i in pred_indices] # Fixed variable name case

#         # 4. Apply Color Correction
#         final_labels = []
#         for i, label in enumerate(pred_labels):
#             corrected = correct_color_errors(model_inputs[i], label)
#             final_labels.append(corrected)

#         # 5. Assemble FEN
#         fen = assemble_fen_from_predictions(final_labels)
        
#         if pov == 'b':
#             fen = black_perspective_fen(fen)
#             turn = 'b'
#         else:
#             turn = 'w'
            
#         final_fen = f"{fen} {turn} KQkq - 0 1"

#         # --- NEW: Encode the board image to send back to UI ---
#         # This allows the user to see exactly what the "Snipper" saw
#         is_success, buffer = cv2.imencode(".jpg", board_viz)
#         if is_success:
#             base64_image = base64.b64encode(buffer).decode('utf-8')
#             cropped_image_data = f"data:image/jpeg;base64,{base64_image}"
#         else:
#             cropped_image_data = None

#         return jsonify({
#             'fen': final_fen,
#             'cropped_image': cropped_image_data # Returning the crop!
#         })

#     except Exception as e:
#         print(e)
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     load_resources()
#     app.run(debug=True, port=5000)



## Code with login and authentication

import tensorflow as tf
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
# Security Key (Change this for production!)
app.config['SECRET_KEY'] = 'chess-vision-secret-key-mvp-2025' 
# Database File (Will be created in project folder)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chessvision.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

MODEL_PATH = Path(r"C:\Users\prajjwal\Desktop\chess-to-fen\Gererate-FEN-notation-from-Chess-Position-Image\Fine_tuned_CNN_Model\chess_model_v4.keras")
LABELS_PATH = Path(r"C:\Users\prajjwal\Desktop\chess-to-fen\Gererate-FEN-notation-from-Chess-Position-Image\labels\class_names.txt")


# Email Config (Your Bot Credentials)
EMAIL_SENDER = "your.bot.email@gmail.com" 
EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"    
EMAIL_RECEIVER = "your.personal@gmail.com"

# --- INIT EXTENSIONS ---
db.init_app(app) # Initialize DB with app

login_manager = LoginManager()
login_manager.login_view = 'login' # Redirect here if user tries to access protected page
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# --- MODEL LOADING & PATCHING ---
MODEL = None
CLASS_NAMES = None

class PatchedRandomContrast(tf.keras.layers.RandomContrast):
    def __init__(self, factor, value_range=None, **kwargs):
        super().__init__(factor, **kwargs)

def load_resources():
    global MODEL, CLASS_NAMES
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        print("--- Loading Model & Labels... ---")
        try:
            MODEL = tf.keras.models.load_model(MODEL_PATH)
        except (ValueError, TypeError) as e:
            print(f"Applying patch... {e}")
            with tf.keras.utils.custom_object_scope({'RandomContrast': PatchedRandomContrast}):
                MODEL = tf.keras.models.load_model(MODEL_PATH)
        CLASS_NAMES = LABELS_PATH.read_text().splitlines()
        print("--- System Ready ---")
    else:
        print(f"ERROR: Could not find model at {MODEL_PATH}")

# --- AUTH ROUTES ---

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', 'error')
        else:
            # Hash password and create user
            hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(email=email, username=username, password=hashed_pw)
            
            try:
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user) # Auto login after signup
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
    # Get last 5 scans for current user
    user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(5).all()
    
    history_data = []
    for scan in user_scans:
        history_data.append({
            'fen': scan.fen,
            'image': scan.image_data, # Base64 string
            'date': scan.timestamp.strftime("%b %d, %H:%M")
        })
    return jsonify(history_data)

# --- MAIN APP LOGIC ---

@app.route('/')
def index():
    # Pass 'current_user' to template to toggle Login/Logout buttons
    return render_template('index.html', user=current_user)

# Helper 1: TTA Voting
def predict_with_voting(model, squares_batch):
    augmented_squares = []
    for sq in squares_batch:
        augmented_squares.append(sq) 
        augmented_squares.append(np.roll(sq, -2, axis=1))
        augmented_squares.append(np.roll(sq, -2, axis=0))
        augmented_squares.append(np.clip(sq * 0.7, 0, 255))
        h, w = 64, 64
        crop = sq[4:60, 4:60]
        aug_zoom = cv2.resize(crop, (64, 64))
        augmented_squares.append(aug_zoom)
    
    big_batch = np.array(augmented_squares)
    preds = model.predict(big_batch, verbose=0)
    num_classes = preds.shape[1]
    reshaped_preds = preds.reshape(64, 5, num_classes)
    
    final_indices = []
    for i in range(64):
        votes = np.argmax(reshaped_preds[i], axis=1)
        winner = stats.mode(votes, keepdims=True).mode[0]
        final_indices.append(winner)
    return final_indices

# Helper 2: Color Correction
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
    if MODEL is None: return jsonify({'error': 'Model not loaded'}), 500
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
        
        # Prediction Pipeline
        pred_indices = predict_with_voting(MODEL, model_inputs)
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

        # Encode image for UI & Database
        is_success, buffer = cv2.imencode(".jpg", board_viz)
        if is_success:
            base64_image = base64.b64encode(buffer).decode('utf-8')
            cropped_image_data = f"data:image/jpeg;base64,{base64_image}"
        else:
            cropped_image_data = None

        # --- SAVE TO DB (If Logged In) ---
        if current_user.is_authenticated:
            try:
                # We save the Base64 string directly. 
                # In production, you'd upload to S3 and save the URL.
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
        msg['Subject'] = f"[ChessVision Report] {tags}"
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
    crop_file = request.files.get('cropped_image') # We need to implement this in JS if we want crop too
    
    orig_bytes = orig_file.read() if orig_file else None
    crop_bytes = crop_file.read() if crop_file else None

    thread = threading.Thread(target=send_email_async, args=(feedback, tags, fen, orig_bytes, crop_bytes))
    thread.start()

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    load_resources()
    # Create DB tables if they don't exist
    with app.app_context():
        db.create_all()
        print("--- Database Connected & Tables Ready ---")
    app.run(debug=True, port=5000)