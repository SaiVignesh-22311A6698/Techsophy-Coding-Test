from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
from app.services.ehr_analyzer import EHRAnalyzer
from app.services.data_ingestion import DataIngestionService
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"], supports_credentials=True)

data_ingestion = DataIngestionService()
ehr_analyzer = EHRAnalyzer()

@app.route("/")
def root():
    return jsonify({
        "message": "EHR Data Quality Auditor API (Flask)",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "upload": "/upload",
            "sample-data": "/sample-data"
        }
    })

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "service": "EHR Data Quality Auditor (Flask)"})

@app.route("/upload", methods=["POST"])
def upload_and_analyze():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        filename = file.filename
        if not (filename.endswith('.csv') or filename.endswith('.json') or filename.endswith('.xlsx')):
            return jsonify({"error": "Unsupported file format. Please upload CSV, JSON, or Excel files."}), 400
        # Read and process the uploaded file
        data = data_ingestion.read_file(file)
        # Analyze
        result = ehr_analyzer.analyze_data(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route("/analyze", methods=["POST"])
def analyze_ehr_data():
    try:
        req_json = request.get_json()
        data = req_json.get('data') if req_json else None
        if not data:
            # Use sample data
            data = data_ingestion.get_sample_data()
        # Analyze
        result = ehr_analyzer.analyze_data(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error in analysis: {str(e)}"}), 500

@app.route("/sample-data")
def get_sample_data():
    try:
        sample_data = data_ingestion.get_sample_data()
        return jsonify({
            "message": "Sample EHR data retrieved successfully",
            "data": sample_data,
            "record_count": len(sample_data) if sample_data else 0
        })
    except Exception as e:
        logger.error(f"Error retrieving sample data: {str(e)}")
        return jsonify({"error": f"Error retrieving sample data: {str(e)}"}), 500

@app.route("/metrics")
def get_quality_metrics():
    return jsonify({
        "completeness_metrics": {
            "required_fields": ["patient_id", "date_of_birth", "gender", "diagnosis"],
            "optional_fields": ["phone", "email", "emergency_contact"],
            "completeness_threshold": 0.85
        },
        "consistency_metrics": {
            "age_validation": {"min": 0, "max": 120},
            "blood_pressure": {"systolic": {"min": 70, "max": 200}, "diastolic": {"min": 40, "max": 130}},
            "heart_rate": {"min": 40, "max": 200},
            "temperature": {"min": 35.0, "max": 42.0}
        },
        "accuracy_metrics": {
            "outlier_detection": "isolation_forest",
            "pattern_matching": "kmeans_clustering",
            "statistical_validation": "z_score_analysis"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True) 