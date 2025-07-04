from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ErrorSeverity(str, Enum):
    """Error severity levels for quality issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorType(str, Enum):
    """Types of data quality errors"""
    MISSING_DATA = "missing_data"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"
    INCONSISTENT_DATA = "inconsistent_data"
    DUPLICATE_RECORD = "duplicate_record"
    ANOMALY_DETECTED = "anomaly_detected"

class QualityDimension(str, Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"

class DataQualityError(BaseModel):
    """Individual data quality error"""
    field_name: str
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    record_id: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class QualityMetric(BaseModel):
    """Quality metric for a specific dimension"""
    dimension: QualityDimension
    score: float = Field(..., ge=0.0, le=1.0)
    total_records: int
    error_count: int
    errors: List[DataQualityError] = []
    details: Dict[str, Any] = {}

class CompletenessMetrics(BaseModel):
    """Completeness analysis results"""
    overall_score: float = Field(..., ge=0.0, le=1.0)
    missing_fields: Dict[str, int] = {}
    required_fields_completeness: float
    optional_fields_completeness: float
    field_completeness: Dict[str, float] = {}

class ConsistencyMetrics(BaseModel):
    """Consistency analysis results"""
    overall_score: float = Field(..., ge=0.0, le=1.0)
    cross_field_errors: List[DataQualityError] = []
    format_errors: List[DataQualityError] = []
    logical_errors: List[DataQualityError] = []

class AccuracyMetrics(BaseModel):
    """Accuracy analysis results"""
    overall_score: float = Field(..., ge=0.0, le=1.0)
    outliers_detected: int
    statistical_errors: List[DataQualityError] = []
    range_violations: List[DataQualityError] = []
    anomaly_scores: Dict[str, float] = {}

class TimelinessMetrics(BaseModel):
    """Timeliness analysis results"""
    overall_score: float = Field(..., ge=0.0, le=1.0)
    data_freshness: float
    update_frequency: Optional[float] = None
    temporal_consistency: float

class OutlierAnalysis(BaseModel):
    """Outlier detection results"""
    total_outliers: int
    outlier_percentage: float
    outlier_records: List[Dict[str, Any]] = []
    statistical_summary: Dict[str, Any] = {}

class PatternAnalysis(BaseModel):
    """Pattern matching results"""
    patterns_detected: List[str] = []
    cluster_analysis: Dict[str, Any] = {}
    data_distribution: Dict[str, Any] = {}

class EHRDataRequest(BaseModel):
    """Request model for EHR data analysis"""
    data: Optional[List[Dict[str, Any]]] = None
    use_sample_data: bool = False
    analysis_options: Dict[str, Any] = Field(default_factory=dict)

class QualityReport(BaseModel):
    """Comprehensive quality report"""
    report_id: str
    timestamp: datetime
    total_records: int
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)
    
    # Detailed metrics
    completeness: CompletenessMetrics
    consistency: ConsistencyMetrics
    accuracy: AccuracyMetrics
    timeliness: TimelinessMetrics
    
    # AI/ML Analysis
    outlier_analysis: OutlierAnalysis
    pattern_analysis: PatternAnalysis
    
    # Summary
    total_errors: int
    critical_errors: int
    high_priority_errors: int
    medium_priority_errors: int
    low_priority_errors: int
    
    # Recommendations
    recommendations: List[str] = []
    priority_actions: List[str] = []
    
    # Metadata
    processing_time: float
    analysis_version: str = "1.0.0"

class EHRSampleData(BaseModel):
    """Sample EHR data structure"""
    patient_id: str
    date_of_birth: str
    gender: str
    diagnosis: str
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    temperature: Optional[float] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    emergency_contact: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    lab_results: Optional[Dict[str, Any]] = None

class ValidationRule(BaseModel):
    """Data validation rule"""
    field_name: str
    rule_type: str
    parameters: Dict[str, Any]
    severity: ErrorSeverity
    description: str

class AnalysisConfig(BaseModel):
    """Configuration for data analysis"""
    completeness_threshold: float = 0.85
    outlier_contamination: float = 0.1
    statistical_confidence: float = 0.95
    enable_ml_analysis: bool = True
    validation_rules: List[ValidationRule] = [] 