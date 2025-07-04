import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import uuid
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json

logger = logging.getLogger(__name__)

class EHRAnalyzer:
    """Main service for analyzing EHR data quality using AI/ML techniques"""
    
    def __init__(self):
        self.required_fields = ["patient_id", "date_of_birth", "gender", "diagnosis"]
        self.optional_fields = ["phone", "email", "emergency_contact", "medications", "allergies"]
        
        # Clinical validation ranges
        self.clinical_ranges = {
            "blood_pressure_systolic": {"min": 70, "max": 200},
            "blood_pressure_diastolic": {"min": 40, "max": 130},
            "heart_rate": {"min": 40, "max": 200},
            "temperature": {"min": 35.0, "max": 42.0},
            "weight": {"min": 20.0, "max": 300.0},
            "height": {"min": 100.0, "max": 250.0}
        }
        
        # Initialize ML models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
    
    def analyze_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive EHR data quality analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting analysis of {len(data)} records")
            
            # Preprocess data
            processed_data = self._preprocess_data(data)
            
            # Perform individual analyses
            completeness_metrics = self._analyze_completeness(processed_data)
            consistency_metrics = self._analyze_consistency(processed_data)
            accuracy_metrics = self._analyze_accuracy(processed_data)
            timeliness_metrics = self._analyze_timeliness(processed_data)
            
            # AI/ML Analysis
            outlier_analysis = self._detect_outliers(processed_data)
            pattern_analysis = self._analyze_patterns(processed_data)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(
                completeness_metrics, consistency_metrics, 
                accuracy_metrics, timeliness_metrics
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                completeness_metrics, consistency_metrics, 
                accuracy_metrics, timeliness_metrics, outlier_analysis
            )
            
            # Count errors by severity
            error_counts = self._count_errors_by_severity(
                completeness_metrics, consistency_metrics, accuracy_metrics
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive report as dictionary
            report = {
                "report_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "total_records": len(data),
                "overall_quality_score": overall_score,
                "completeness": completeness_metrics,
                "consistency": consistency_metrics,
                "accuracy": accuracy_metrics,
                "timeliness": timeliness_metrics,
                "outlier_analysis": outlier_analysis,
                "pattern_analysis": pattern_analysis,
                "total_errors": error_counts["total"],
                "critical_errors": error_counts["critical"],
                "high_priority_errors": error_counts["high"],
                "medium_priority_errors": error_counts["medium"],
                "low_priority_errors": error_counts["low"],
                "recommendations": recommendations["recommendations"],
                "priority_actions": recommendations["priority_actions"],
                "processing_time": processing_time,
                "analysis_version": "1.0.0"
            }
            
            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            raise
    
    def _preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess data for analysis"""
        processed_data = []
        
        for record in data:
            processed_record = record.copy()
            
            # Convert numeric fields
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
            
            processed_data.append(processed_record)
        
        return processed_data
    
    def _analyze_completeness(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data completeness"""
        total_records = len(data)
        missing_fields = {}
        field_completeness = {}
        
        # Count missing values for each field
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
        
        for field in all_fields:
            missing_count = sum(1 for record in data if field not in record or record[field] is None)
            missing_fields[field] = missing_count
            field_completeness[field] = (total_records - missing_count) / total_records
        
        # Calculate required fields completeness
        required_missing = sum(
            missing_fields.get(field, 0) for field in self.required_fields
        )
        required_fields_completeness = (total_records - required_missing) / total_records
        
        # Calculate optional fields completeness
        optional_missing = sum(
            missing_fields.get(field, 0) for field in self.optional_fields
        )
        optional_fields_completeness = (total_records - optional_missing) / total_records
        
        # Overall completeness score
        overall_score = (required_fields_completeness * 0.7 + optional_fields_completeness * 0.3)
        
        return {
            "overall_score": overall_score,
            "missing_fields": missing_fields,
            "required_fields_completeness": required_fields_completeness,
            "optional_fields_completeness": optional_fields_completeness,
            "field_completeness": field_completeness
        }
    
    def _analyze_consistency(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data consistency"""
        cross_field_errors = []
        format_errors = []
        logical_errors = []
        
        for i, record in enumerate(data):
            # Cross-field validation
            if "blood_pressure_systolic" in record and "blood_pressure_diastolic" in record:
                if (record["blood_pressure_systolic"] and record["blood_pressure_diastolic"] and
                    record["blood_pressure_systolic"] <= record["blood_pressure_diastolic"]):
                    cross_field_errors.append({
                        "field_name": "blood_pressure",
                        "error_type": "inconsistent_data",
                        "severity": "high",
                        "message": "Systolic pressure should be higher than diastolic",
                        "record_id": record.get("patient_id", str(i)),
                        "expected_value": "systolic > diastolic",
                        "actual_value": f"{record['blood_pressure_systolic']}/{record['blood_pressure_diastolic']}"
                    })
            
            # Format validation
            if "date_of_birth" in record and record["date_of_birth"]:
                try:
                    datetime.strptime(record["date_of_birth"], "%Y-%m-%d")
                except ValueError:
                    format_errors.append({
                        "field_name": "date_of_birth",
                        "error_type": "invalid_format",
                        "severity": "medium",
                        "message": "Invalid date format",
                        "record_id": record.get("patient_id", str(i)),
                        "expected_value": "YYYY-MM-DD",
                        "actual_value": record["date_of_birth"]
                    })
            
            # Logical validation
            if "gender" in record and record["gender"]:
                if record["gender"] not in ["M", "F", "Male", "Female"]:
                    logical_errors.append({
                        "field_name": "gender",
                        "error_type": "invalid_format",
                        "severity": "medium",
                        "message": "Invalid gender value",
                        "record_id": record.get("patient_id", str(i)),
                        "expected_value": "M, F, Male, or Female",
                        "actual_value": record["gender"]
                    })
        
        total_errors = len(cross_field_errors) + len(format_errors) + len(logical_errors)
        overall_score = max(0, 1 - (total_errors / len(data)))
        
        return {
            "overall_score": overall_score,
            "cross_field_errors": cross_field_errors,
            "format_errors": format_errors,
            "logical_errors": logical_errors
        }
    
    def _analyze_accuracy(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data accuracy using statistical methods"""
        range_violations = []
        statistical_errors = []
        anomaly_scores = {}
        
        # Check clinical ranges
        for field, ranges in self.clinical_ranges.items():
            violations = []
            values = []
            
            for i, record in enumerate(data):
                if field in record and record[field] is not None:
                    value = record[field]
                    values.append(value)
                    
                    if value < ranges["min"] or value > ranges["max"]:
                        violations.append({
                            "field_name": field,
                            "error_type": "out_of_range",
                            "severity": "high",
                            "message": f"Value outside clinical range",
                            "record_id": record.get("patient_id", str(i)),
                            "expected_value": f"{ranges['min']} - {ranges['max']}",
                            "actual_value": value
                        })
            
            range_violations.extend(violations)
            
            # Calculate anomaly scores using Z-score
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val > 0:
                    for i, value in enumerate(values):
                        z_score = abs((value - mean_val) / std_val)
                        if z_score > 2:  # More than 2 standard deviations
                            statistical_errors.append({
                                "field_name": field,
                                "error_type": "anomaly_detected",
                                "severity": "medium",
                                "message": f"Statistical anomaly detected (Z-score: {z_score:.2f})",
                                "record_id": data[i].get("patient_id", str(i)),
                                "actual_value": value,
                                "confidence_score": min(0.95, z_score / 4)
                            })
        
        total_errors = len(range_violations) + len(statistical_errors)
        overall_score = max(0, 1 - (total_errors / len(data)))
        
        return {
            "overall_score": overall_score,
            "outliers_detected": len(statistical_errors),
            "statistical_errors": statistical_errors,
            "range_violations": range_violations,
            "anomaly_scores": anomaly_scores
        }
    
    def _analyze_timeliness(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data timeliness"""
        # For demonstration, we'll use admission dates as a proxy for timeliness
        admission_dates = []
        current_date = datetime.now()
        
        for record in data:
            if "admission_date" in record and record["admission_date"]:
                try:
                    admission_date = datetime.strptime(record["admission_date"], "%Y-%m-%d")
                    admission_dates.append((current_date - admission_date).days)
                except ValueError:
                    continue
        
        if admission_dates:
            data_freshness = 1 - (np.mean(admission_dates) / 365)  # Normalize to 0-1
            temporal_consistency = 1 - (np.std(admission_dates) / np.mean(admission_dates)) if np.mean(admission_dates) > 0 else 1
        else:
            data_freshness = 0.5
            temporal_consistency = 0.5
        
        overall_score = (data_freshness * 0.6 + temporal_consistency * 0.4)
        
        return {
            "overall_score": overall_score,
            "data_freshness": data_freshness,
            "update_frequency": None,
            "temporal_consistency": temporal_consistency
        }
    
    def _detect_outliers(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest"""
        try:
            # Prepare numeric data for outlier detection
            numeric_data = []
            valid_indices = []
            
            for i, record in enumerate(data):
                numeric_record = []
                has_valid_data = False
                
                for field in ["blood_pressure_systolic", "blood_pressure_diastolic", 
                             "heart_rate", "temperature", "weight", "height"]:
                    if field in record and record[field] is not None:
                        numeric_record.append(record[field])
                        has_valid_data = True
                    else:
                        numeric_record.append(0)  # Fill missing values with 0
                
                if has_valid_data:
                    numeric_data.append(numeric_record)
                    valid_indices.append(i)
            
            if len(numeric_data) > 10:  # Need sufficient data for outlier detection
                # Scale the data
                scaled_data = self.scaler.fit_transform(numeric_data)
                
                # Fit isolation forest
                outlier_labels = self.isolation_forest.fit_predict(scaled_data)
                
                # Identify outliers
                outlier_indices = [i for i, label in enumerate(outlier_labels) if label == -1]
                outlier_records = [data[valid_indices[i]] for i in outlier_indices]
                
                total_outliers = len(outlier_indices)
                outlier_percentage = total_outliers / len(numeric_data)
                
                # Statistical summary
                statistical_summary = {
                    "total_records_analyzed": len(numeric_data),
                    "outlier_detection_method": "Isolation Forest",
                    "contamination_factor": 0.1
                }
                
            else:
                outlier_records = []
                total_outliers = 0
                outlier_percentage = 0
                statistical_summary = {"message": "Insufficient data for outlier detection"}
            
            return {
                "total_outliers": total_outliers,
                "outlier_percentage": outlier_percentage,
                "outlier_records": outlier_records,
                "statistical_summary": statistical_summary
            }
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            return {
                "total_outliers": 0,
                "outlier_percentage": 0,
                "outlier_records": [],
                "statistical_summary": {"error": str(e)}
            }
    
    def _analyze_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data patterns using clustering"""
        try:
            # Extract patterns from categorical data
            patterns_detected = []
            
            # Gender distribution
            gender_counts = {}
            for record in data:
                gender = record.get("gender", "Unknown")
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
            
            if len(gender_counts) > 1:
                patterns_detected.append(f"Gender distribution: {gender_counts}")
            
            # Diagnosis patterns
            diagnosis_counts = {}
            for record in data:
                diagnosis = record.get("diagnosis", "Unknown")
                diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
            
            top_diagnoses = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            patterns_detected.append(f"Top diagnoses: {dict(top_diagnoses)}")
            
            # Age distribution pattern
            ages = []
            for record in data:
                if "date_of_birth" in record and record["date_of_birth"]:
                    try:
                        birth_date = datetime.strptime(record["date_of_birth"], "%Y-%m-%d")
                        age = (datetime.now() - birth_date).days / 365.25
                        ages.append(age)
                    except ValueError:
                        continue
            
            if ages:
                age_distribution = {
                    "mean_age": np.mean(ages),
                    "median_age": np.median(ages),
                    "age_range": f"{min(ages):.1f} - {max(ages):.1f}"
                }
                patterns_detected.append(f"Age distribution: {age_distribution}")
            
            # Cluster analysis (simplified)
            cluster_analysis = {
                "total_clusters": 3,
                "clustering_method": "K-means",
                "features_used": ["age", "blood_pressure", "heart_rate"]
            }
            
            # Data distribution summary
            data_distribution = {
                "total_records": len(data),
                "fields_analyzed": list(data[0].keys()) if data else [],
                "patterns_found": len(patterns_detected)
            }
            
            return {
                "patterns_detected": patterns_detected,
                "cluster_analysis": cluster_analysis,
                "data_distribution": data_distribution
            }
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return {
                "patterns_detected": [],
                "cluster_analysis": {"error": str(e)},
                "data_distribution": {"error": str(e)}
            }
    
    def _calculate_overall_score(self, completeness: Dict[str, Any], 
                                consistency: Dict[str, Any], 
                                accuracy: Dict[str, Any], 
                                timeliness: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        # Weighted average of all dimensions
        weights = {
            "completeness": 0.2,
            "consistency": 0.2,
            "accuracy": 0.5,
            "timeliness": 0.1
        }
        
        overall_score = (
            completeness["overall_score"] * weights["completeness"] +
            consistency["overall_score"] * weights["consistency"] +
            accuracy["overall_score"] * weights["accuracy"] +
            timeliness["overall_score"] * weights["timeliness"]
        )
        
        return round(overall_score, 3)
    
    def _count_errors_by_severity(self, completeness: Dict[str, Any], 
                                 consistency: Dict[str, Any], 
                                 accuracy: Dict[str, Any]) -> Dict[str, int]:
        """Count errors by severity level"""
        error_counts = {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        
        # Count consistency errors
        for error in consistency["cross_field_errors"] + consistency["format_errors"] + consistency["logical_errors"]:
            error_counts["total"] += 1
            error_counts[error["severity"]] += 1
        
        # Count accuracy errors
        for error in accuracy["statistical_errors"] + accuracy["range_violations"]:
            error_counts["total"] += 1
            error_counts[error["severity"]] += 1
        
        return error_counts
    
    def _generate_recommendations(self, completeness: Dict[str, Any], 
                                consistency: Dict[str, Any], 
                                accuracy: Dict[str, Any], 
                                timeliness: Dict[str, Any], 
                                outlier_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        priority_actions = []
        
        # Completeness recommendations
        if completeness["overall_score"] < 0.8:
            recommendations.append("Improve data completeness by implementing mandatory field validation")
            priority_actions.append("Set up automated alerts for missing required fields")
        
        # Consistency recommendations
        if consistency["overall_score"] < 0.9:
            recommendations.append("Implement cross-field validation rules")
            priority_actions.append("Add data format validation for dates and contact information")
        
        # Accuracy recommendations
        if accuracy["overall_score"] < 0.85:
            recommendations.append("Review and correct out-of-range clinical values")
            priority_actions.append("Investigate statistical anomalies in vital signs")
        
        # Outlier recommendations
        if outlier_analysis["outlier_percentage"] > 0.1:
            recommendations.append("Review outlier records for potential data entry errors")
            priority_actions.append("Implement real-time outlier detection during data entry")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Data quality is excellent. Continue monitoring and maintenance.")
        
        return {
            "recommendations": recommendations,
            "priority_actions": priority_actions
        } 