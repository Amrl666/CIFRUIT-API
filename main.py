import os
import io
import numpy as np
from PIL import Image
import mysql.connector
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Comprehensive class mapping for fruits and their maturity stages
class_mapping = {'buahnaga_busuk': 0, 'buahnaga_matang': 1, 'buahnaga_mentah': 2, 
                'jeruk_busuk': 3, 'jeruk_matang': 4, 'jeruk_mentah': 5, 
                'pepaya_mentah': 6, 'pepaya_busuk': 7, 'pepaya_matang': 8, 
                'pisang_busuk': 9, 'pisang_matang': 10, 'pisang_mentah': 11, 
                'rambutan_mentah': 12, 'rambutan_busuk': 13, 'rambutan_matang': 14}

# Database Configuration - Use environment variables in production
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '34.101.36.201'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root123'),
    'database': os.getenv('DB_NAME', 'buah_db')
}

# Model and Database Initialization
model = None
db = None
cursor = None

try:
    logger.info("Attempting to load model from path: model.h5")
    
    # Check if model file exists
    if not os.path.exists("model.h5"):
        raise FileNotFoundError("Model file 'model.h5' not found in current directory")

    def load_model_for_fruit():
        model_path = "model.h5"  # Adjust the path as needed
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model with custom objects if necessary
        model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
        return model
    
    # Load the model
    model = load_model_for_fruit()
    
    # Verify model is loaded correctly
    if model is None:
        raise ValueError("Model loading resulted in None object")
    
    logger.info(f"Model successfully loaded. Summary: {model.summary()}")

    # Establish database connection
    db = mysql.connector.connect(**DB_CONFIG)
    cursor = db.cursor()
    
    # Check if the connection is successful
    if db.is_connected():
        logger.info("Database connection established successfully.")
        init_database()  # Initialize the database after successful connection
    else:
        logger.error("Database connection failed.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
def preprocess_image(img):
    try:
        img = img.resize((224, 224), Image.LANCZOS)
        img = img.convert('RGB')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Image Preprocessing Error: {e}")
        raise Exception("Gambar tidak valid atau format salah")

def predict_image(img):
    try:
        logger.info(f"Image details: size={img.size}, mode={img.mode}")
        
        img_array = preprocess_image(img)
        
        logger.info(f"Preprocessed array shape: {img_array.shape}")
        
        predictions = model.predict(img_array)
        
        if predictions is None or len(predictions[0]) == 0:
            raise ValueError("No predictions generated")
        
        predicted_class_index = np.argmax (predictions)
        predicted_class = list(class_mapping.keys())[list(class_mapping.values()).index(predicted_class_index)]
        confidence = predictions[0][predicted_class_index]
        
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Detailed Prediction Error: {e}", exc_info=True)
        raise Exception("Prediksi gagal")

def save_prediction_to_db(image_name, predicted_class, confidence, top_predictions=None):
    try:
        if cursor is None or db is None:
            logger.error("Database connection or cursor is not initialized.")
            return
        
        query = """
        INSERT INTO predictions 
        (image_name, predicted_class, confidence, top_predictions) 
        VALUES (%s, %s, %s, %s)
        """
        
        top_predictions_json = json.dumps(top_predictions or {})
        
        cursor.execute(query, (image_name, predicted_class, float(confidence), top_predictions_json))
        db.commit()
    
    except mysql.connector.Error as err:
        logger.error(f"Database Error: {err}")
        db.rollback()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({
            "error": "No file uploaded",
            "details": "Ensure you're sending a file with key 'file'"
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            "error": "No file selected",
            "details": "Please choose a valid image file"
        }), 400
    
    try:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not file.filename.lower().split('.')[-1] in allowed_extensions:
            return jsonify({
                "error": "Invalid file type",
                "details": f"Allowed extensions: {', '.join(allowed_extensions)}"
            }), 400
        
        image_obj = Image.open(io.BytesIO(file.read()))
        
        predicted_class, confidence = predict_image(image_obj)
        
        confidence_percentage = confidence * 100
        
        if "matang" in predicted_class:
            recommendation = "Buah matang, segera konsumsi untuk rasa terbaik."
        elif "mentah" in predicted_class:
            recommendation = "Buah mentah, simpan hingga matang sebelum dikonsumsi."
        elif "busuk" in predicted_class:
            recommendation = "Buah busuk, disarankan untuk dibuang atau digunakan sebagai pupuk kompos."
        else:
            recommendation = "Tidak ada rekomendasi khusus."
        
        save_prediction_to_db(file.filename, predicted_class, confidence_percentage, {})
        
        return jsonify({
            "image_name": file.filename,
            "predicted_class": predicted_class,
            "confidence": f"{confidence_percentage:.2f}%",
            "recommendation": recommendation
        })
    
    except Exception as e:
        logger.error(f"Prediction Endpoint Error: {e}", exc_info=True)
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    try:
        model_status = "Loaded" if model is not None else "Not Loaded"
        db_status = "Connected" if db is not None else "Disconnected"
        
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "model_path": os.path.abspath("model.h5") if os.path.exists("model.h5") else "Model file not found",
            "database_status": db_status,
            "supported_classes": list(class_mapping.keys()),
            "tensorflow_version": tf.__version__,
            "python_version": sys.version
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Welcome to Fruit Maturity Prediction API",
        "available_routes": ["/predict"],
        "model_classes": list(class_mapping.keys())
    })

def init_database():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_name VARCHAR(255),
        predicted_class VARCHAR(50),
        confidence FLOAT,
        top_predictions JSON,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    try:
        cursor.execute(create_table_query)
        db.commit()
        logger.info("Database table initialized successfully")
    except mysql.connector.Error as err:
        logger.error(f"Database initialization error: {err}")

if cursor:
    init_database()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) 