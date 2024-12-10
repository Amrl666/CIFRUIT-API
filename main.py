import os
import io
import sys
import json
import uuid
import shutil
import atexit
import logging
import numpy as np
import mysql.connector
import tensorflow as tf
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi Konstanta
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Pemetaan Kelas
class_mapping = {
    'buahnaga_busuk': 0, 'buahnaga_matang': 1, 'buahnaga_mentah': 2, 
    'jeruk_busuk': 3, 'jeruk_matang': 4, 'jeruk_mentah': 5, 
    'pepaya_mentah': 6, 'pepaya_busuk': 7, 'pepaya_matang': 8, 
    'pisang_busuk': 9, 'pisang_matang': 10, 'pisang_mentah': 11, 
    'rambutan_mentah': 12, 'rambutan_busuk': 13, 'rambutan_matang': 14
}

# Konfigurasi Database
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '34.101.36.201'),
    'user': os.getenv('DB_USER', 'cifruit_user'),
    'password': os.getenv('DB_PASSWORD', 'cifruit123'),
    'database': os.getenv('DB_NAME', 'buah_db'),
    'connection_timeout': 10,
    'pool_name': "mypool",
    'pool_size': 5
}

# Variabel Global
global db, cursor, model
db = None
cursor = None
model = None

def get_server_status():
    """Mendapatkan status server secara komprehensif"""
    try:
        # Cek koneksi database
        db_connected = check_db_connection()
        
        # Cek status model
        model_loaded = model is not None
        
        # Cek ruang disk
        total, used, free = shutil.disk_usage('/')
        
        return {
            "database_status": db_connected,
            "model_status": model_loaded,
            "disk_space": {
                "total": total // (2**30),  # Convert to GB
                "used": used // (2**30),
                "free": free // (2**30),
                "free_percentage": (free / total) * 100
            },
            "timestamp": datetime.now().isoformat(),
            "supported_classes": list(class_mapping.keys())
        }
    except Exception as e:
        logger.error(f"Server status check error: {e}")
        return {
            "database_status": False,
            "model_status": False,
            "error": str(e)
        }

def load_model_for_fruit():
    """Memuat model machine learning"""
    try:
        model_path = "model.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
        )
        logger.info(f"Model successfully loaded. Summary: {model.summary()}")
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise

def init_database():
    """Inisialisasi tabel database"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        prediction_id VARCHAR(36) UNIQUE,
        image_name VARCHAR(255),
        predicted_class VARCHAR(50),
        confidence FLOAT,
        top_predictions JSON,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    try:
        logger.info("Attempting to create table...")
        cursor.execute(create_table_query)
        db.commit()
        logger.info("Database table initialized successfully")
    except mysql.connector.Error as err:
        logger.error(f"Database initialization error: {err}")
        print(f"Detailed Error: {err}")
        raise

def check_db_connection():
    """Memeriksa dan mengembalikan koneksi database"""
    global db, cursor
    try:
        if db is None or not db.is_connected():
            logger.info("Reconnecting to database...")
            db = mysql.connector.connect(**DB_CONFIG)
            cursor = db.cursor(dictionary=True)
            
            # Tambahkan pengecekan tambahan
            cursor.execute("SELECT DATABASE()")
            current_db = cursor.fetchone()
            logger.info(f"Connected to database: {current_db}")
        return True
    except mysql.connector.Error as err:
        logger.error(f"Database reconnection failed: {err}")
        print(f"Detailed Connection Error: {err}")
        return False

def close_db_connection():
    """Menutup koneksi database"""
    global db, cursor
    try:
        if cursor:
            cursor.close()
        if db and db.is_connected():
            db.close()
        logger.info("Database connection closed.")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")

def preprocess_image(img):
    """Mempersiapkan gambar untuk prediksi"""
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
    """Melakukan prediksi pada gambar"""
    try:
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)

        predicted_class_index = np.argmax(predictions)
        predicted_class = list(class_mapping.keys())[list(class_mapping.values()).index(predicted_class_index)]
        confidence = predictions[0][predicted_class_index]

        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Prediction Error: {e}", exc_info=True)
        raise Exception("Prediksi gagal")

def save_prediction_to_db(prediction_id, image_name, predicted_class, confidence, top_predictions=None):
    """Menyimpan hasil prediksi ke database dengan ID unik"""
    try:
        if not check_db_connection():
            logger.error("Cannot save prediction - database not connected")
            return

        query = """
        INSERT INTO predictions 
        (prediction_id, image_name, predicted_class, confidence, top_predictions) 
        VALUES (%s, %s, %s, %s, %s)
        """

        top_predictions_json = json.dumps(top_predictions or {})

        cursor.execute(query, (
            prediction_id, 
            image_name, 
            predicted_class, 
            float(confidence), 
            top_predictions_json
        ))
        db.commit()
        logger.info(f"Prediction saved: {prediction_id} - {predicted_class}")

    except mysql.connector.Error as err:
        logger.error(f"Database Save Error: {err}")
        try:
            db.rollback()
        except:
            pass
    except Exception as e:
        logger.error(f"Unexpected error saving prediction: {e}")

# Inisialisasi Awal
try:
    # Muat model
    model = load_model_for_fruit()

    # Sambungkan database
    db = mysql.connector.connect(**DB_CONFIG)
    cursor = db.cursor(dictionary=True)

    # Inisialisasi tabel
    init_database()
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    sys.exit(1)

# Registrasi penutupan koneksi database
atexit.register(close_db_connection)

# Route Prediksi
@app.route("/predict", methods=["POST"])
def predict():
    # Periksa koneksi database
    server_status = get_server_status()
    if not server_status['database_status']:
        return jsonify({
            "error": "Database connection failed",
            "server_status": server_status,
            "details": "Unable to connect to the database"
        }), 500

    # Pemeriksaan file
    if 'file' not in request.files:
        return jsonify({
            "error": "No file uploaded",
            "server_status": server_status
        }), 400

    file = request.files['file']

    # Validasi ukuran dan ekstensi file
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return jsonify({
            "error": "File size exceeds the limit of 10MB",
            "server_status": server_status
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "File type not allowed",
            "server_status": server_status
        }), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        predicted_class, confidence = predict_image(img)
        confidence_percentage = confidence * 100
        
        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())

        if "matang" in predicted_class:
            recommendation = "Buah matang, segera konsumsi untuk rasa terbaik."
        elif "mentah" in predicted_class:
            recommendation = "Buah mentah, simpan hingga matang sebelum dikonsumsi."
        elif "busuk" in predicted_class:
            recommendation = "Buah busuk, disarankan untuk dibuang atau digunakan sebagai pupuk kompos."
        else:
            recommendation = "Tidak ada rekomendasi khusus."
        
        # Simpan prediksi dengan ID baru
        save_prediction_to_db(
            prediction_id=prediction_id,
            image_name=file.filename, 
            predicted_class=predicted_class, 
            confidence=confidence
        )

        return jsonify({
            "prediction_id": prediction_id,
            "image_name": file.filename,
            "predicted_class": predicted_class,
            "confidence": f"{confidence_percentage:.2f}%",
            "recommendation": recommendation,
            "server_status": server_status
        }), 200
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({
            "error": "Prediction failed", 
            "details": str(e),
            "server_status": server_status
        }), 500

def allowed_file(filename):
    """Memeriksa apakah ekstensi file diizinkan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route Status Database
@app.route("/database-status", methods=["GET"])
def database_status():
    try:
        server_status = get_server_status()
        if server_status['database_status']:
            return jsonify(server_status), 200
        else:
            return jsonify(server_status), 500
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return jsonify({
            "status": "error", 
            "details": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    try:
        server_status = get_server_status()
        
        return jsonify({
            "status": "healthy" if server_status['database_status'] and server_status['model_status'] else "unhealthy",
            "details": server_status
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route("/history", methods=["GET"])
def get_prediction_history():
    server_status = get_server_status()
    if not server_status['database_status']:
        return jsonify({
            "error": "Database connection failed",
            "server_status": server_status,
            "details": "Unable to connect to the database"
        }), 500
    
    try:
        # Mengambil parameter limit dari query string, default 10 jika tidak ada
        limit = request.args.get('limit', default=10, type=int)
        
        query = """
        SELECT prediction_id, image_name, predicted_class, confidence, prediction_time
        FROM predictions
        ORDER BY prediction_time DESC
        LIMIT %s
        """
        
        cursor.execute(query, (limit,))
        history = cursor.fetchall()
        
        # Mengubah format datetime menjadi string untuk JSON serialization
        for item in history:
            item['prediction_time'] = item['prediction_time'].isoformat()
            item['confidence'] = f"{item['confidence']*100:.2f}%"
        
        return jsonify({
            "history": history,
            "count": len(history),
            "server_status": server_status
        }), 200
    
    except mysql.connector.Error as err:
        logger.error(f"Database query error: {err}")
        return jsonify({
            "error": "Database query failed", 
            "details": str(err),
            "server_status": server_status
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error in history route: {e}")
        return jsonify({
            "error": "Internal server error", 
            "details": str(e),
            "server_status": server_status
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)