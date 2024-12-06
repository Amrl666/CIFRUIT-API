<<<<<<< HEAD
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Comprehensive class mapping for fruits and their maturity stages
class_mapping = {'buahnaga_busuk': 0, 'buahnaga_matang': 1, 'buahnaga_mentah': 2, 
                'jeruk_busuk': 3, 'jeruk_matang': 4, 'jeruk_mentah': 5, 
                'pepaya mentah': 6, 'pepaya_busuk': 7, 'pepaya_matang': 8, 
                'pisang_busuk': 9, 'pisang_matang': 10, 'pisang_mentah': 11, 
                'rambutan mentah': 12, 'rambutan_busuk': 13, 'rambutan_matang': 14}

# Database Configuration - Use environment variables in production
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'buah_db')
}

# Model and Database Initialization
try:
    # Load pre-trained model with EfficientNet architecture
    model = tf.keras.models.load_model("model.h5")
    
    # Establish database connection
    db = mysql.connector.connect(**DB_CONFIG)
    cursor = db.cursor()
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    model = None
    db = None
    cursor = None

def preprocess_image(img):
    """
    Praproses gambar agar sesuai dengan model MobileNetV2
    Args:
        img (PIL.Image): Input gambar
    Returns:
        numpy.ndarray: Array gambar yang sudah diproses
    """
    try:
        # Ubah ukuran gambar menjadi 224x224
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Konversi ke mode RGB
        img = img.convert('RGB')
        
        # Konversi gambar menjadi array numpy
        img_array = image.img_to_array(img)
        
        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Gunakan preprocess_input dari MobileNetV2
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        logger.error(f"Image Preprocessing Error: {e}")
        raise Exception("Gambar tidak valid atau format salah")

def predict_image(img):
    """
    Prediksi kelas buah dan tingkat kepercayaannya
    Args:
        img (PIL.Image): Input gambar
    Returns:
        tuple: (predicted class, confidence)
    """
    try:
        # Praproses gambar
        img_array = preprocess_image(img)
        
        # Prediksi menggunakan model
        predictions = model.predict(img_array)
        
        # Ambil indeks kelas dengan confidence tertinggi
        predicted_class_index = np.argmax(predictions)
        
        # Ambil nama kelas berdasarkan mapping
        predicted_class = list(class_mapping.keys())[list(class_mapping.values()).index(predicted_class_index)]
        
        # Ambil confidence
        confidence = predictions[0][predicted_class_index]
        
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise Exception("Prediksi gagal")


def save_prediction_to_db(image_name, predicted_class, confidence, top_predictions=None):
    """
    Simpan hasil prediksi ke database
    Args:
        image_name (str): Nama file gambar
        predicted_class (str): Kelas yang diprediksi
        confidence (float): Tingkat kepercayaan
        top_predictions (dict): (Opsional) Prediksi teratas lainnya
    """
    try:
        query = """
        INSERT INTO predictions 
        (image_name, predicted_class, confidence, top_predictions) 
        VALUES (%s, %s, %s, %s)
        """
        
        # Jika `top_predictions` kosong, simpan JSON kosong
        top_predictions_json = json.dumps(top_predictions or {})
        
        cursor.execute(query, (image_name, predicted_class, float(confidence), top_predictions_json))
        db.commit()
    
    except mysql.connector.Error as err:
        logger.error(f"Database Error: {err}")
        db.rollback()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint utama untuk prediksi
    Returns:
        dict: Hasil prediksi beserta rekomendasi
    """
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "File tidak dipilih"}), 400
    
    try:
        # Membaca gambar dari file yang diunggah
        image_obj = Image.open(io.BytesIO(file.read()))
        
        # Prediksi gambar
        predicted_class, confidence = predict_image(image_obj)
        
        # Konversi confidence ke bentuk persentase
        confidence_percentage = confidence * 100
        
        # Tambahkan rekomendasi berdasarkan prediksi
        if "matang" in predicted_class:
            recommendation = "Buah matang, segera konsumsi untuk rasa terbaik."
        elif "mentah" in predicted_class:
            recommendation = "Buah mentah, simpan hingga matang sebelum dikonsumsi."
        elif "busuk" in predicted_class:
            recommendation = "Buah busuk, disarankan untuk dibuang atau digunakan sebagai pupuk kompos."
        else:
            recommendation = "Tidak ada rekomendasi khusus."
        
        # Menyimpan hasil ke database (opsional)
        save_prediction_to_db(file.filename, predicted_class, confidence_percentage, {})  # Kosongkan `top_predictions`
        
        # Mengembalikan hasil prediksi dan rekomendasi
        return jsonify({
            "image_name": file.filename,
            "predicted_class": predicted_class,
            "confidence": f"{confidence_percentage:.2f}%",
            "recommendation": recommendation
        })
    
    except Exception as e:
        logger.error(f"Error di endpoint prediksi: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/", methods=["GET"])
def root():
    """Root endpoint with basic API information"""
    return jsonify({
        "message": "Welcome to Fruit Maturity Prediction API",
        "available_routes": ["/predict"],
        "model_classes": list(class_mapping.keys())
    })

# Optional database initialization script
def init_database():
    """Create predictions table if not exists"""
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

# Call database initialization if needed
if cursor:
    init_database()

if __name__ == "__main__":
=======
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Comprehensive class mapping for fruits and their maturity stages
class_mapping = {'buahnaga_busuk': 0, 'buahnaga_matang': 1, 'buahnaga_mentah': 2, 
                'jeruk_busuk': 3, 'jeruk_matang': 4, 'jeruk_mentah': 5, 
                'pepaya mentah': 6, 'pepaya_busuk': 7, 'pepaya_matang': 8, 
                'pisang_busuk': 9, 'pisang_matang': 10, 'pisang_mentah': 11, 
                'rambutan mentah': 12, 'rambutan_busuk': 13, 'rambutan_matang': 14}

# Database Configuration - Use environment variables in production
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'buah_db')
}

# Model and Database Initialization
try:
    # Load pre-trained model with EfficientNet architecture
    model = tf.keras.models.load_model("model.h5")
    
    # Establish database connection
    db = mysql.connector.connect(**DB_CONFIG)
    cursor = db.cursor()
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    model = None
    db = None
    cursor = None

def preprocess_image(img):
    """
    Praproses gambar agar sesuai dengan model MobileNetV2
    Args:
        img (PIL.Image): Input gambar
    Returns:
        numpy.ndarray: Array gambar yang sudah diproses
    """
    try:
        # Ubah ukuran gambar menjadi 224x224
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Konversi ke mode RGB
        img = img.convert('RGB')
        
        # Konversi gambar menjadi array numpy
        img_array = image.img_to_array(img)
        
        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Gunakan preprocess_input dari MobileNetV2
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        logger.error(f"Image Preprocessing Error: {e}")
        raise Exception("Gambar tidak valid atau format salah")

def predict_image(img):
    """
    Prediksi kelas buah dan tingkat kepercayaannya
    Args:
        img (PIL.Image): Input gambar
    Returns:
        tuple: (predicted class, confidence)
    """
    try:
        # Praproses gambar
        img_array = preprocess_image(img)
        
        # Prediksi menggunakan model
        predictions = model.predict(img_array)
        
        # Ambil indeks kelas dengan confidence tertinggi
        predicted_class_index = np.argmax(predictions)
        
        # Ambil nama kelas berdasarkan mapping
        predicted_class = list(class_mapping.keys())[list(class_mapping.values()).index(predicted_class_index)]
        
        # Ambil confidence
        confidence = predictions[0][predicted_class_index]
        
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise Exception("Prediksi gagal")


def save_prediction_to_db(image_name, predicted_class, confidence, top_predictions=None):
    """
    Simpan hasil prediksi ke database
    Args:
        image_name (str): Nama file gambar
        predicted_class (str): Kelas yang diprediksi
        confidence (float): Tingkat kepercayaan
        top_predictions (dict): (Opsional) Prediksi teratas lainnya
    """
    try:
        query = """
        INSERT INTO predictions 
        (image_name, predicted_class, confidence, top_predictions) 
        VALUES (%s, %s, %s, %s)
        """
        
        # Jika `top_predictions` kosong, simpan JSON kosong
        top_predictions_json = json.dumps(top_predictions or {})
        
        cursor.execute(query, (image_name, predicted_class, float(confidence), top_predictions_json))
        db.commit()
    
    except mysql.connector.Error as err:
        logger.error(f"Database Error: {err}")
        db.rollback()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint utama untuk prediksi
    Returns:
        dict: Hasil prediksi beserta rekomendasi
    """
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "File tidak dipilih"}), 400
    
    try:
        # Membaca gambar dari file yang diunggah
        image_obj = Image.open(io.BytesIO(file.read()))
        
        # Prediksi gambar
        predicted_class, confidence = predict_image(image_obj)
        
        # Konversi confidence ke bentuk persentase
        confidence_percentage = confidence * 100
        
        # Tambahkan rekomendasi berdasarkan prediksi
        if "matang" in predicted_class:
            recommendation = "Buah matang, segera konsumsi untuk rasa terbaik."
        elif "mentah" in predicted_class:
            recommendation = "Buah mentah, simpan hingga matang sebelum dikonsumsi."
        elif "busuk" in predicted_class:
            recommendation = "Buah busuk, disarankan untuk dibuang atau digunakan sebagai pupuk kompos."
        else:
            recommendation = "Tidak ada rekomendasi khusus."
        
        # Menyimpan hasil ke database (opsional)
        save_prediction_to_db(file.filename, predicted_class, confidence_percentage, {})  # Kosongkan `top_predictions`
        
        # Mengembalikan hasil prediksi dan rekomendasi
        return jsonify({
            "image_name": file.filename,
            "predicted_class": predicted_class,
            "confidence": f"{confidence_percentage:.2f}%",
            "recommendation": recommendation
        })
    
    except Exception as e:
        logger.error(f"Error di endpoint prediksi: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/", methods=["GET"])
def root():
    """Root endpoint with basic API information"""
    return jsonify({
        "message": "Welcome to Fruit Maturity Prediction API",
        "available_routes": ["/predict"],
        "model_classes": list(class_mapping.keys())
    })

# Optional database initialization script
def init_database():
    """Create predictions table if not exists"""
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

# Call database initialization if needed
if cursor:
    init_database()

if __name__ == "__main__":
>>>>>>> 4dd6801 (first commit)
    app.run(host='0.0.0.0', port=5000)