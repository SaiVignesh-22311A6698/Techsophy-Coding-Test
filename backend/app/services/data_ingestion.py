import pandas as pd
import json
import io
import logging
from typing import List, Dict, Any, Optional
import random
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class DataIngestionService:
    """Service for ingesting and processing EHR data from various sources"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.xlsx']
        
    def read_file(self, file) -> List[Dict[str, Any]]:
        """
        Read and parse uploaded file based on its format
        """
        try:
            content = file.read()
            
            if file.filename.endswith('.csv'):
                return self._parse_csv(content)
            elif file.filename.endswith('.json'):
                return self._parse_json(content)
            elif file.filename.endswith('.xlsx'):
                return self._parse_excel(content)
            else:
                raise ValueError(f"Unsupported file format: {file.filename}")
                
        except Exception as e:
            logger.error(f"Error reading file {file.filename}: {str(e)}")
            raise
    
    def _parse_csv(self, content: bytes) -> List[Dict[str, Any]]:
        """Parse CSV content"""
        try:
            df = pd.read_csv(io.BytesIO(content))
            df = df.replace({np.nan: None})  # Replace NaN with None for JSON compatibility
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error parsing CSV: {str(e)}")
            raise
    
    def _parse_json(self, content: bytes) -> List[Dict[str, Any]]:
        """Parse JSON content"""
        try:
            data = json.loads(content.decode('utf-8'))
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'records' in data:
                return data['records']
            else:
                return [data]
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            raise
    
    def _parse_excel(self, content: bytes) -> List[Dict[str, Any]]:
        """Parse Excel content"""
        try:
            df = pd.read_excel(io.BytesIO(content))
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error parsing Excel: {str(e)}")
            raise
    
    def get_sample_data(self) -> List[Dict[str, Any]]:
        """
        Generate comprehensive sample EHR data for demonstration
        """
        sample_data = []
        
        # Generate 100 sample records
        for i in range(100):
            record = self._generate_sample_record(i + 1)
            sample_data.append(record)
        
        logger.info(f"Generated {len(sample_data)} sample EHR records")
        return sample_data
    
    def _generate_sample_record(self, record_id: int) -> Dict[str, Any]:
        """Generate a single sample EHR record"""
        
        # Basic patient information
        patient_id = f"P{record_id:06d}"
        gender = random.choice(['M', 'F'])
        
        # Generate realistic date of birth (18-90 years old)
        age = random.randint(18, 90)
        birth_year = datetime.now().year - age
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)
        date_of_birth = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"
        
        # Common diagnoses
        diagnoses = [
            "Hypertension", "Diabetes Type 2", "Asthma", "Heart Disease",
            "Arthritis", "Depression", "Anxiety", "Obesity", "COPD",
            "Chronic Kidney Disease", "Cancer", "Stroke", "Migraine"
        ]
        diagnosis = random.choice(diagnoses)
        
        # Vital signs with realistic ranges
        systolic = random.randint(90, 180)
        diastolic = random.randint(60, 110)
        heart_rate = random.randint(50, 100)
        temperature = round(random.uniform(36.0, 38.5), 1)
        weight = round(random.uniform(45.0, 120.0), 1)
        height = round(random.uniform(150.0, 200.0), 1)
        
        # Contact information (sometimes missing for completeness testing)
        phone = f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}" if random.random() > 0.1 else None
        email = f"patient{record_id}@example.com" if random.random() > 0.15 else None
        emergency_contact = f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}" if random.random() > 0.2 else None
        
        # Dates
        admission_date = None
        discharge_date = None
        if random.random() > 0.3:  # 70% have admission dates
            admission_date = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
            if random.random() > 0.4:  # 60% have discharge dates
                discharge_date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
        
        # Medications
        medications = []
        if random.random() > 0.2:
            med_options = [
                "Aspirin", "Metformin", "Lisinopril", "Atorvastatin",
                "Amlodipine", "Omeprazole", "Albuterol", "Sertraline"
            ]
            num_meds = random.randint(1, 3)
            medications = random.sample(med_options, min(num_meds, len(med_options)))
        
        # Allergies
        allergies = []
        if random.random() > 0.7:
            allergy_options = ["Penicillin", "Peanuts", "Latex", "Sulfa drugs", "Iodine"]
            num_allergies = random.randint(1, 2)
            allergies = random.sample(allergy_options, min(num_allergies, len(allergy_options)))
        
        # Lab results
        lab_results = {}
        if random.random() > 0.3:
            lab_results = {
                "glucose": round(random.uniform(70, 200), 1),
                "cholesterol": round(random.uniform(120, 300), 1),
                "creatinine": round(random.uniform(0.5, 2.0), 2),
                "hemoglobin": round(random.uniform(10, 18), 1)
            }
        
        # Introduce some data quality issues for testing
        if random.random() < 0.05:  # 5% chance of invalid blood pressure
            systolic = random.randint(200, 300)  # Unrealistic high
        
        if random.random() < 0.03:  # 3% chance of invalid temperature
            temperature = round(random.uniform(30.0, 35.0), 1)  # Too low
        
        if random.random() < 0.04:  # 4% chance of missing critical field
            diagnosis = None
        
        record = {
            "patient_id": patient_id,
            "date_of_birth": date_of_birth,
            "gender": gender,
            "diagnosis": diagnosis,
            "blood_pressure_systolic": systolic,
            "blood_pressure_diastolic": diastolic,
            "heart_rate": heart_rate,
            "temperature": temperature,
            "weight": weight,
            "height": height,
            "phone": phone,
            "email": email,
            "emergency_contact": emergency_contact,
            "admission_date": admission_date,
            "discharge_date": discharge_date,
            "medications": medications,
            "allergies": allergies,
            "lab_results": lab_results
        }
        
        return record
    
    def validate_data_structure(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the structure of ingested data
        """
        if not data:
            return {"valid": False, "errors": ["No data provided"]}
        
        validation_result = {
            "valid": True,
            "total_records": len(data),
            "errors": [],
            "warnings": []
        }
        
        # Check if all records have required fields
        required_fields = ["patient_id", "date_of_birth", "gender", "diagnosis"]
        
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                validation_result["errors"].append(f"Record {i} is not a dictionary")
                validation_result["valid"] = False
                continue
            
            for field in required_fields:
                if field not in record:
                    validation_result["errors"].append(f"Record {i} missing required field: {field}")
                    validation_result["valid"] = False
        
        return validation_result
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess data for analysis
        """
        processed_data = []
        
        for record in data:
            processed_record = record.copy()
            
            # Convert string numbers to appropriate types
            numeric_fields = [
                "blood_pressure_systolic", "blood_pressure_diastolic",
                "heart_rate", "temperature", "weight", "height"
            ]
            
            for field in numeric_fields:
                if field in processed_record and processed_record[field] is not None:
                    try:
                        if isinstance(processed_record[field], str):
                            processed_record[field] = float(processed_record[field])
                    except (ValueError, TypeError):
                        processed_record[field] = None
            
            # Standardize date formats
            date_fields = ["date_of_birth", "admission_date", "discharge_date"]
            for field in date_fields:
                if field in processed_record and processed_record[field]:
                    try:
                        # Ensure date format is consistent
                        if isinstance(processed_record[field], str):
                            # Basic date validation
                            datetime.strptime(processed_record[field], "%Y-%m-%d")
                    except ValueError:
                        processed_record[field] = None
            
            processed_data.append(processed_record)
        
        return processed_data 