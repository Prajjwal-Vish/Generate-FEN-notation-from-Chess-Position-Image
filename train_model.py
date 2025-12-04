### This is the code that i used to train a new CNN model form scratch to identify chess pieces

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import zipfile
import shutil
from pathlib import Path

# 1. CONFIGURATION & SETUP
from google.colab import drive
drive.mount('/content/drive')

# --- USER PATHS (Update these if your folder names differ) ---
# Folder on Drive where your data lives
DRIVE_FOLDER = Path('/content/drive/My Drive/chess-to-fen-project-dataset/')

# Input files
ZIP_FILE = DRIVE_FOLDER / 'raw_data2.zip'
LABELS_FILE = DRIVE_FOLDER / 'class_names.txt'

# Output files
OUTPUT_TFLITE_MODEL = DRIVE_FOLDER / 'chess_model_v5.tflite'
OUTPUT_KERAS_MODEL = DRIVE_FOLDER / 'chess_model_v5.keras'

# Training Config
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50

# ==========================================
# 2. DATA EXTRACTION
# ==========================================
print("\n--- Step 2: Extracting Data ---")
LOCAL_DATA_ROOT = Path('/content/dataset')

# Clean up previous runs
if LOCAL_DATA_ROOT.exists():
    shutil.rmtree(LOCAL_DATA_ROOT)
LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Unzip
print(f"Unzipping {ZIP_FILE}...")
try:
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(LOCAL_DATA_ROOT)
except FileNotFoundError:
    print(f"ERROR: Could not find {ZIP_FILE}. Check your Drive path!")
    raise

# Find the specific subfolder containing the class folders
DATA_DIR = None
for root, dirs, files in os.walk(LOCAL_DATA_ROOT):
    # We look for a folder that has subdirectories (the classes)
    if len(dirs) > 1:
        DATA_DIR = Path(root)
        break

print(f"Data Source: {DATA_DIR}")

# ==========================================
# 3. LOAD LABELS & DATASET
# ==========================================
print("\n--- Step 3: Loading Datasets ---")

# Read strict class order from text file
if LABELS_FILE.exists():
    with open(LABELS_FILE, 'r') as f:
        CUSTOM_CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Forcing Class Order: {CUSTOM_CLASS_NAMES}")
else:
    raise FileNotFoundError("class_labels.txt not found! This is required to match your app.")

# Load Training Data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CUSTOM_CLASS_NAMES, # Crucial: Enforces index mapping
    label_mode='int'
)

# Load Validation Data
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CUSTOM_CLASS_NAMES,
    label_mode='int'
)

# Performance Optimization
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 4. BUILD CUSTOM MODEL (For Deployment)
# ==========================================
print("\n--- Step 4: Building Custom Architecture ---")

def build_deployable_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),

        # --- A. Augmentation (Runs on GPU during training) ---
        # Solving the "Mobile Screenshot" issues:
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),    # Fixes display differences
        layers.RandomBrightness(0.2),  # Fixes "Night Mode" issues

        # --- B. Preprocessing ---
        # Normalize pixel values (0-255 -> 0-1) inside the model
        layers.Rescaling(1./255),

        # --- C. Feature Extraction (Lightweight CNN) ---
        # Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(), # 64 -> 32

        # Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(), # 32 -> 16

        # Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(), # 16 -> 8

        # Block 4 (Spatial Features)
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        # No pooling here - we want to keep the 8x8 grid for color context

        # --- D. Classification ---
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5), # Prevents overfitting on small data

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_deployable_model(len(CUSTOM_CLASS_NAMES))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 5. TRAINING
# ==========================================
print("\n--- Step 5: Training ---")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('temp_best.keras', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 6. EXPORT FOR DEPLOYMENT
# ==========================================
print("\n--- Step 6: Converting to TFLite ---")

# 1. Save Keras Backup
model.save(OUTPUT_KERAS_MODEL)
print(f"Backup Keras model saved to: {OUTPUT_KERAS_MODEL}")

# 2. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save TFLite to Drive
with open(OUTPUT_TFLITE_MODEL, 'wb') as f:
    f.write(tflite_model)

print(f"\nSUCCESS! Deployment model ready.")
print(f"Location: {OUTPUT_TFLITE_MODEL}")
print("Size: {:.2f} MB".format(len(tflite_model) / 1024 / 1024))
