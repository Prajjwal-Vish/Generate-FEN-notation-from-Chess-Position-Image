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





# import numpy as np
# import cv2
# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
# from pathlib import Path
# import io
# from scipy import stats
# import base64 
# import smtplib
# import threading
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.image import MIMEImage
# import os
# from dotenv import load_dotenv

# # --- SMART IMPORT BLOCK (CRITICAL FOR WINDOWS/RENDER COMPATIBILITY) ---
# # This must be the ONLY place we import tflite logic.
# try:
#     # 1. Try Lightweight Runtime (For Render/Linux)
#     import tflite_runtime.interpreter as tflite
#     print("--- Using TFLite Runtime (Lightweight) ---")
# except ImportError:
#     # 2. Fallback to Full TensorFlow (For Local Windows)
#     try:
#         import tensorflow.lite as tflite
#         print("--- Using Full TensorFlow Lite (Local Fallback) ---")
#     except ImportError:
#         print("CRITICAL ERROR: Neither 'tflite_runtime' nor 'tensorflow' is installed.")
#         print("Please run: pip install tensorflow")
# # ---------------------------------------------------------------------

# # Load environment variables locally
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
# app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chess-vision-secret-key-mvp-2025')

# # Database: Checks for Render's DATABASE_URL, falls back to local SQLite
# database_url = os.environ.get('DATABASE_URL', 'sqlite:///chessvision.db')
# if database_url and database_url.startswith("postgres://"):
#     database_url = database_url.replace("postgres://", "postgresql://", 1)

# app.config['SQLALCHEMY_DATABASE_URI'] = database_url
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # --- PATH CONFIGURATION ---
# BASE_DIR = Path(__file__).resolve().parent

# # Point to the TFLite model
# MODEL_PATH = BASE_DIR / "Fine_tuned_CNN_Model" / "chess_model_v4.tflite"
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

# # --- TFLITE MODEL LOADING ---
# INTERPRETER = None
# INPUT_DETAILS = None
# OUTPUT_DETAILS = None
# CLASS_NAMES = None

# def load_resources():
#     global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, CLASS_NAMES
    
#     # --- DEBUGGING: PRINT FILE SYSTEM ---
#     print("\n" + "="*30)
#     print("🔍 DEBUGGING PATHS")
#     print(f"Looking for model at: {MODEL_PATH}")
    
#     if not MODEL_PATH.exists():
#         print(f"❌ ERROR: Model file NOT found at {MODEL_PATH}")
#         # Print what IS in the folder to help debug casing issues
#         parent_dir = MODEL_PATH.parent
#         if parent_dir.exists():
#             print(f"Contents of {parent_dir}:")
#             for item in parent_dir.iterdir():
#                 print(f"   - {item.name}")
#         else:
#              print(f"❌ The folder {parent_dir} does not exist either!")
#     else:
#         print(f"✅ Model File Found!")
#     print("="*30 + "\n")
#     # ------------------------------------

#     if MODEL_PATH.exists() and LABELS_PATH.exists():
#         print(f"--- Loading TFLite Model... ---")
#         try:
#             # Load TFLite Interpreter
#             # Note: We use 'tflite' here, which maps to whichever library was successfully imported at the top
#             INTERPRETER = tflite.Interpreter(model_path=str(MODEL_PATH))
#             INTERPRETER.allocate_tensors()
            
#             # Get input/output details for inference
#             INPUT_DETAILS = INTERPRETER.get_input_details()
#             OUTPUT_DETAILS = INTERPRETER.get_output_details()
            
#             CLASS_NAMES = LABELS_PATH.read_text().splitlines()
#             print("--- System Ready (TFLite Mode) ---")
#         except Exception as e:
#             print(f"CRITICAL ERROR loading model: {e}")
#             import traceback
#             traceback.print_exc()
#     else:
#         print(f"ERROR: Could not find model/labels.")

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

# # Helper: TFLite Prediction Wrapper
# def tflite_predict(interpreter, input_data):
#     """
#     Runs inference on a batch of images using TFLite.
#     input_data shape: (Batch_Size, 64, 64, 3)
#     """
#     input_index = INPUT_DETAILS[0]['index']
#     output_index = OUTPUT_DETAILS[0]['index']
    
#     predictions = []
    
#     # Loop through each image in the batch
#     for i in range(len(input_data)):
#         # 1. Prepare single input (1, 64, 64, 3)
#         # Ensure it's float32 as expected by the model
#         img = input_data[i:i+1].astype(np.float32) 
        
#         # 2. Set Tensor
#         interpreter.set_tensor(input_index, img)
        
#         # 3. Invoke
#         interpreter.invoke()
        
#         # 4. Get Result
#         output = interpreter.get_tensor(output_index)
#         predictions.append(output[0])
        
#     return np.array(predictions)

# def predict_with_voting(interpreter, squares_batch):
#     augmented_squares = []
#     for sq in squares_batch:
#         # Standard TTA augmentations
#         augmented_squares.append(sq) 
#         augmented_squares.append(np.roll(sq, -2, axis=1))
#         augmented_squares.append(np.roll(sq, -2, axis=0))
#         augmented_squares.append(np.clip(sq * 0.7, 0, 255))
        
#         h, w = 64, 64
#         crop = sq[4:60, 4:60]
#         aug_zoom = cv2.resize(crop, (64, 64))
#         augmented_squares.append(aug_zoom)
    
#     big_batch = np.array(augmented_squares)
    
#     # USE TFLITE INFERENCE HERE
#     preds = tflite_predict(interpreter, big_batch)
    
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
#     if INTERPRETER is None: return jsonify({'error': 'Model not loaded'}), 500
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
        
#         # Pass INTERPRETER instead of MODEL
#         pred_indices = predict_with_voting(INTERPRETER, model_inputs)
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

#         # Save to History if logged in
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

# # --- EMAIL LOGIC ---
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











# import numpy as np
# import cv2
# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
# from pathlib import Path
# import io
# from scipy import stats
# import base64 
# import smtplib
# import threading
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.image import MIMEImage
# import os
# from dotenv import load_dotenv

# # --- BULLETPROOF IMPORT BLOCK ---
# tflite = None # Initialize as None to prevent NameError

# try:
#     # 1. Try Lightweight Runtime (Preferred for Render)
#     import tflite_runtime.interpreter as tflite
#     print("--- ✅ SUCCESS: Using TFLite Runtime ---")
# except ImportError as e1:
#     print(f"--- ⚠️ TFLite Runtime Import Failed: {e1} ---")
#     try:
#         # 2. Fallback to Full TensorFlow (For Local/Dev)
#         import tensorflow.lite as tflite
#         print("--- ✅ SUCCESS: Using Full TensorFlow Lite ---")
#     except ImportError as e2:
#         print(f"--- ❌ CRITICAL: TensorFlow Import Failed: {e2} ---")
#         print("--- No valid TFLite library found. Model loading will fail. ---")
# # --------------------------------

# load_dotenv()

# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, login_user, login_required, logout_user, current_user
# from werkzeug.security import generate_password_hash, check_password_hash
# from models import db, User, Scan 

# try:
#     from chessboard_snipper import process_image
#     from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen
# except ImportError:
#     print("CRITICAL: Missing helper modules.")

# app = Flask(__name__)

# app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chess-vision-secret-key-mvp-2025')

# database_url = os.environ.get('DATABASE_URL', 'sqlite:///chessvision.db')
# if database_url and database_url.startswith("postgres://"):
#     database_url = database_url.replace("postgres://", "postgresql://", 1)

# app.config['SQLALCHEMY_DATABASE_URI'] = database_url
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = BASE_DIR / "Fine_tuned_CNN_Model" / "chess_model_v5.tflite" # Updated to v5
# LABELS_PATH = BASE_DIR / "labels" / "class_names.txt"

# EMAIL_SENDER = os.environ.get('EMAIL_SENDER')
# EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
# EMAIL_RECEIVER = os.environ.get('EMAIL_RECEIVER')

# db.init_app(app) 
# login_manager = LoginManager()
# login_manager.login_view = 'login' 
# login_manager.init_app(app)

# @login_manager.user_loader
# def load_user(id):
#     return User.query.get(int(id))

# INTERPRETER = None
# INPUT_DETAILS = None
# OUTPUT_DETAILS = None
# CLASS_NAMES = None

# def load_resources():
#     global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, CLASS_NAMES
    
#     print("\n" + "="*30)
#     print("🔍 DEPLOYMENT DEBUGGER")
#     print(f"Base Directory: {BASE_DIR}")
    
#     # Check Files
#     model_exists = MODEL_PATH.exists()
#     labels_exists = LABELS_PATH.exists()
    
#     if model_exists: print(f"✅ Model File Found: {MODEL_PATH}")
#     else: print(f"❌ Model NOT Found at: {MODEL_PATH}")
    
#     if labels_exists: print(f"✅ Labels File Found: {LABELS_PATH}")
#     else: print(f"❌ Labels NOT Found at: {LABELS_PATH}")

#     # Check Library
#     if tflite is None:
#         print("❌ CRITICAL: TFLite Library is NOT loaded. Cannot initialize interpreter.")
#     else:
#         print(f"✅ TFLite Library is loaded: {tflite}")

#     print("="*30 + "\n")

#     if model_exists and labels_exists and tflite is not None:
#         try:
#             INTERPRETER = tflite.Interpreter(model_path=str(MODEL_PATH))
#             INTERPRETER.allocate_tensors()
#             INPUT_DETAILS = INTERPRETER.get_input_details()
#             OUTPUT_DETAILS = INTERPRETER.get_output_details()
#             CLASS_NAMES = LABELS_PATH.read_text().splitlines()
#             print("--- System Ready (TFLite Mode) ---")
#         except Exception as e:
#             print(f"CRITICAL ERROR loading model: {e}")
#             import traceback
#             traceback.print_exc()

# # ... [Routes for Signup, Login, Logout remain unchanged] ...
# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         username = request.form.get('username')
#         password = request.form.get('password')
#         user = User.query.filter_by(email=email).first()
#         if user: flash('Email exists.', 'error')
#         else:
#             new_user = User(email=email, username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
#             db.session.add(new_user); db.session.commit(); login_user(new_user)
#             return redirect(url_for('index'))
#     return render_template('signup.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         user = User.query.filter_by(email=request.form.get('email')).first()
#         if user and check_password_hash(user.password, request.form.get('password')):
#             login_user(user, remember=True); return redirect(url_for('index'))
#         else: flash('Invalid credentials.', 'error')
#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout(): logout_user(); return redirect(url_for('index'))

# @app.route('/api/history')
# @login_required
# def get_history():
#     try:
#         scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(5).all()
#         return jsonify([{'fen': s.fen, 'image': s.image_data, 'date': s.timestamp.strftime("%b %d")} for s in scans])
#     except: return jsonify([])

# @app.route('/')
# def index(): return render_template('index.html', user=current_user)

# # ... [Helper functions tflite_predict, voting, color correction remain unchanged] ...
# def tflite_predict(interpreter, input_data):
#     # Ensure interpreter is loaded
#     if interpreter is None: raise ValueError("Model Interpreter not loaded")
#     input_idx = INPUT_DETAILS[0]['index']
#     output_idx = OUTPUT_DETAILS[0]['index']
#     preds = []
#     for i in range(len(input_data)):
#         img = input_data[i:i+1].astype(np.float32)
#         interpreter.set_tensor(input_idx, img)
#         interpreter.invoke()
#         preds.append(interpreter.get_tensor(output_idx)[0])
#     return np.array(preds)

# def predict_with_voting(interpreter, squares_batch):
#     # ... (Same augmentation logic) ...
#     augmented = []
#     for sq in squares_batch:
#         augmented.append(sq)
#         augmented.append(np.roll(sq, -2, axis=1))
#         augmented.append(np.roll(sq, -2, axis=0))
#         augmented.append(np.clip(sq * 0.7, 0, 255))
#         augmented.append(cv2.resize(sq[4:60, 4:60], (64, 64)))
    
#     preds = tflite_predict(interpreter, np.array(augmented))
#     # Vote logic
#     reshaped = preds.reshape(64, 5, preds.shape[1])
#     return [stats.mode(np.argmax(reshaped[i], axis=1), keepdims=True).mode[0] for i in range(64)]

# def correct_color_errors(img, label):
#     # ... (Same color logic) ...
#     if "empty" in label: return label
#     if img.dtype != np.uint8: img = img.astype(np.uint8)
#     gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape
#     center = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
#     corners = np.mean([gray[0:10,0:10], gray[0:10,w-10:w], gray[h-10:h,0:10], gray[h-10:h,w-10:w]])
#     if center - corners > 30 and "dark" in label: return label.replace("dark", "light")
#     if center - corners < -30 and "light" in label: return label.replace("light", "dark")
#     return label

# @app.route('/predict', methods=['POST'])
# def predict():
#     if INTERPRETER is None: return jsonify({'error': 'AI Model not loaded on server'}), 500
#     # ... (Same prediction flow) ...
#     try:
#         file = request.files['file']
#         img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#         processed = process_image(img)
#         if not processed: return jsonify({'error': 'Board not found'}), 400
        
#         inputs, viz, _ = processed
#         indices = predict_with_voting(INTERPRETER, inputs)
#         labels = [correct_color_errors(inputs[i], CLASS_NAMES[idx]) for i, idx in enumerate(indices)]
        
#         fen = assemble_fen_from_predictions(labels)
#         if request.form.get('pov') == 'b': fen = black_perspective_fen(fen)
        
#         b64_img = base64.b64encode(cv2.imencode(".jpg", viz)[1]).decode('utf-8')
        
#         if current_user.is_authenticated:
#             db.session.add(Scan(fen=f"{fen} w KQkq - 0 1", image_data=f"data:image/jpeg;base64,{b64_img}", user_id=current_user.id))
#             db.session.commit()
            
#         return jsonify({'fen': f"{fen} w KQkq - 0 1", 'cropped_image': f"data:image/jpeg;base64,{b64_img}"})
#     except Exception as e: return jsonify({'error': str(e)}), 500

# # ... [Email Logic Same as Before] ...
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

# with app.app_context(): db.create_all()
# load_resources()
# if __name__ == '__main__': app.run(debug=True, port=5000)







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

# ===========================
#  TFLITE SAFE IMPORT BLOCK
# ===========================
tflite = None
try:
    import tflite_runtime.interpreter as tflite
    print("✔ Using TFLite Runtime")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("✔ Using TensorFlow Lite Fallback")
    except ImportError:
        print("❌ CRITICAL: No TFLite library available!")

# Load environment variables
load_dotenv()

# ===========================
#  DATABASE + AUTH IMPORTS
# ===========================
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Scan 



# ===========================
#  FLASK APP INITIALIZATION
# ===========================
app = Flask(__name__)

# Import helper modules AFTER app initialization (Render fix)
try:
    from chessboard_snipper import process_image
    from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen
    print("✅ Helper modules loaded successfully.")
except Exception as e:
    print(f"❌ Helper module import failed: {e}")


# Render fix: cannot write to /app/instance
import tempfile
app.instance_path = tempfile.mkdtemp()

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chess-vision-secret-key-mvp')

# ===========================
#  DATABASE CONFIG
# ===========================
database_url = os.environ.get("DATABASE_URL")

if database_url:
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
else:
    database_url = "sqlite:///local.db"   # fallback for local testing

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Login Manager
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# ===========================
#  PATHS FOR MODEL + LABELS
# ===========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Fine_tuned_CNN_Model" / "chess_model_v5.tflite"
LABELS_PATH = BASE_DIR / "labels" / "class_names.txt"

EMAIL_SENDER = os.environ.get('EMAIL_SENDER')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
EMAIL_RECEIVER = os.environ.get('EMAIL_RECEIVER')

# ===========================
#  LOAD MODEL + LABELS
# ===========================
INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None
CLASS_NAMES = None

def load_resources():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, CLASS_NAMES

    print("\n===============================")
    print("🔍 RESOURCE LOADER DEBUG")
    print(f"Model path: {MODEL_PATH}")
    print(f"Labels path: {LABELS_PATH}")

    if not MODEL_PATH.exists():
        print("❌ Model not found!")
    if not LABELS_PATH.exists():
        print("❌ Labels not found!")

    if tflite is None:
        print("❌ No TFLite backend loaded!")
        return

    try:
        INTERPRETER = tflite.Interpreter(model_path=str(MODEL_PATH))
        INTERPRETER.allocate_tensors()
        INPUT_DETAILS = INTERPRETER.get_input_details()
        OUTPUT_DETAILS = INTERPRETER.get_output_details()
        CLASS_NAMES = LABELS_PATH.read_text().splitlines()
        print("✔ Model loaded successfully")
    except Exception as e:
        print("❌ Model load error:", e)

# ===========================
#  AUTH ROUTES
# ===========================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return render_template('signup.html')

        hashed = generate_password_hash(password, method='pbkdf2:sha256')
        user = User(email=email, username=username, password=hashed)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('index'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('index'))
        flash("Invalid email or password.", "error")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ===========================
#  HISTORY API
# ===========================
@app.route('/api/history')
@login_required
def get_history():
    scans = Scan.query.filter_by(user_id=current_user.id)\
                      .order_by(Scan.timestamp.desc())\
                      .limit(5).all()

    return jsonify([
        {
            'fen': s.fen,
            'image': s.image_data,
            'date': s.timestamp.strftime("%b %d, %H:%M")
        }
        for s in scans
    ])

# ===========================
#  MAIN ROUTE
# ===========================
@app.route('/')
def index():
    return render_template('index.html', user=current_user)

# ===========================
#  AI INFERENCE HELPERS
# ===========================
def tflite_predict(interpreter, input_data):
    if interpreter is None:
        raise ValueError("Model interpreter not loaded")

    input_index = INPUT_DETAILS[0]['index']
    output_index = OUTPUT_DETAILS[0]['index']

    preds = []
    for i in range(len(input_data)):
        img = input_data[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        preds.append(interpreter.get_tensor(output_index)[0])

    return np.array(preds)

def predict_with_voting(interpreter, squares):
    augmented = []
    for sq in squares:
        augmented.append(sq)
        augmented.append(np.roll(sq, -2, axis=1))
        augmented.append(np.roll(sq, -2, axis=0))
        augmented.append(np.clip(sq * 0.7, 0, 255))
        augmented.append(cv2.resize(sq[4:60, 4:60], (64, 64)))

    preds = tflite_predict(interpreter, np.array(augmented))
    reshaped = preds.reshape(64, 5, preds.shape[1])

    final_idx = []
    for i in range(64):
        votes = np.argmax(reshaped[i], axis=1)
        final_idx.append(stats.mode(votes, keepdims=True).mode[0])

    return final_idx

# ===========================
#  PREDICTION ROUTE
# ===========================
@app.route('/predict', methods=['POST'])
def predict():
    if INTERPRETER is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img_bytes = request.files['file'].read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    processed = process_image(img)

    if not processed:
        return jsonify({'error': 'Chessboard not detected'}), 400

    squares, viz, _ = processed
    indices = predict_with_voting(INTERPRETER, squares)
        
    from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen

    labels = [CLASS_NAMES[i] for i in indices]
    fen = assemble_fen_from_predictions(labels)

    if request.form.get('pov') == 'b':
        fen = black_perspective_fen(fen)

    b64 = base64.b64encode(cv2.imencode(".jpg", viz)[1]).decode("utf-8")

    if current_user.is_authenticated:
        scan = Scan(fen=fen, image_data=f"data:image/jpeg;base64,{b64}", user_id=current_user.id)
        db.session.add(scan)
        db.session.commit()

    return jsonify({"fen": fen, "cropped_image": f"data:image/jpeg;base64,{b64}"})


# ===========================
#  EMAIL HANDLER
# ===========================
def send_email_async(text, tags, fen, orig_bytes, crop_bytes):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"[SnapFen Report] {tags}"

        msg.attach(MIMEText(f"<h3>Feedback</h3><p>{text}</p><p>FEN: {fen}</p>", 'html'))

        if orig_bytes:
            msg.attach(MIMEImage(orig_bytes, name="original.png"))
        if crop_bytes:
            msg.attach(MIMEImage(crop_bytes, name="crop.png"))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email error:", e)

@app.route('/report_issue', methods=['POST'])
def report_issue():
    tags = request.form.get('tags')
    text = request.form.get('feedback')
    fen = request.form.get('fen')
    
    orig = request.files.get('original_image')
    crop = request.files.get('cropped_image')

    threading.Thread(
        target=send_email_async,
        args=(text, tags, fen, orig.read() if orig else None, crop.read() if crop else None)
    ).start()

    return jsonify({"status": "success"})

# ===========================
#  APP STARTUP
# ===========================
load_resources()

@app.route('/debug-files')     ## route to print the directory structure on render
def debug_files():
    import os
    base = os.path.dirname(__file__)
    tree = []

    for root, dirs, files in os.walk(base):
        level = root.replace(base, "").count(os.sep)
        indent = " " * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            tree.append(f"{subindent}{f}")

    return "<pre>" + "\n".join(tree) + "</pre>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
