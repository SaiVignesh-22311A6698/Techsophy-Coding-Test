# Electronic Health Record (EHR) Data Quality Auditor

A comprehensive tool for analyzing EHR data quality, detecting anomalies, and generating detailed reports on data completeness and accuracy.

## Features

### 🧠 AI/ML Capabilities
- **Outlier Detection**: Advanced statistical methods to identify anomalous data points, with a detailed outlier table showing affected records and fields
- **Pattern Matching**: Machine learning algorithms to detect data patterns and inconsistencies
- **Data Validation Rules**: Configurable validation engine with clinical knowledge base
- **Completeness Scoring**: Intelligent scoring system for data completeness assessment

### 📊 Data Quality Metrics & Visualizations
- **Completeness Analysis**: Missing data detection and scoring
- **Consistency Checks**: Cross-field validation and logical consistency
- **Accuracy Assessment**: Statistical validation against expected ranges
- **Timeliness Evaluation**: Data freshness and update frequency analysis
- **Interactive Dashboard**: Real-time data quality metrics with:
  - **Quality Score Cards**: Visual indicators for completeness, consistency, accuracy
  - **Stacked Error Bar Chart**: Error types per field, with clickable bars for details
  - **Outlier Table**: Scrollable, detailed table of outlier records and values
  - **Diagnosis Donut Chart**: Interactive, professional chart with tooltips, legend, and 'Other' grouping
  - **Info Icons & Tooltips**: Every section/table has an info icon with a clear explanation
  - **Download CSV**: Export filtered data and outlier lists as CSV
  - **Export as PDF**: Download a full visual report of the dashboard

### 🏗️ Modular Architecture
- **Data Ingestion Module**: Handles multiple EHR data formats (CSV, JSON, Excel)
- **Validation Rules Engine**: Configurable validation framework
- **Error Detection Module**: AI-powered anomaly detection
- **Reporting Module**: Comprehensive quality reports generation
- 
![Dashboard Screenshot](dashboard.png)

## Technology Stack

### Frontend
- **HTML, JavaScript, Chart.js, Tailwind CSS** for a modern, responsive UI

### Backend
- **Python Flask** for RESTful API
- **Pandas & NumPy** for data processing
- **Scikit-learn** for machine learning

## Project Structure

```
Techsophy/
├── frontend/                 # Frontend application (HTML/JS)
├── backend/                  # Python Flask backend
│   ├── app/
│   │   ├── models/         # Data models
│   │   ├── services/       # Business logic
├── data/                   # Sample EHR data and test files
```

## Getting Started

### Prerequisites
- Node.js 18+ (for future frontend expansion)
- Python 3.9+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SaiVignesh-22311A6698/Techsophy-Coding-Test
cd Techsophy
```

2. **Setup Backend**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

3. **Open Frontend** Open New Terminal
```bash
cd frontend
python -m http.server 3000
```

4. **Access the Application**
- Frontend: http://localhost:3000 (if served)
- Backend API: http://localhost:8000

## Usage

1. **Upload Data**: Click 'Upload Data', select a CSV/JSON/Excel file, and click 'Upload & Analyze'.
2. **Analyze Sample Data**: Click 'Analyze Sample Data' to use built-in demo data.
3. **Review Results**:
   - **Quality Score Cards**: See overall, completeness, consistency, and accuracy scores.
   - **Error Analysis**: Stacked bar chart shows error types per field. Click bars for details.
   - **Outlier Detection**: Table lists outlier records and their values.
   - **Diagnosis Distribution**: Donut chart shows most common diagnoses, with tooltips and legend.
   - **Recommendations**: AI-generated suggestions for improving data quality.
4. **Export**:
   - **Export as PDF**: Download a full visual report of the dashboard.
   - **Download CSV**: Export filtered data or outlier lists as CSV.

## New Visualizations & User Experience

- **Professional, Modern UI**: Clean layout, accessible color palette, responsive design.
- **Info Icons & Tooltips**: Every table/chart has an info icon with a clear explanation for users.
- **Detailed Outlier Table**: Instantly see which records are outliers and why.
- **Interactive Diagnosis Chart**: Hover for details, legend shows counts and percentages, minor categories grouped as 'Other'.
- **Stacked Error Bar Chart**: Visualize error types per field, click for more info.
- **Download & Export**: Get your results as CSV or PDF for further analysis or sharing.

## Example Data

- Use the provided `data/sample_ehr_data_50.csv` for a realistic demo.
## My Original Github
- https://github.com/saivignesh-balne
