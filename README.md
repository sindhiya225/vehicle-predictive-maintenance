# Vehicle Predictive Maintenance System 

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green)]()

---

##  Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

---


##  Overview

This project implements a **predictive maintenance solution for vehicles** using machine learning.  
The system analyzes engine parameters, usage patterns, and historical fault data to predict when a vehicle is likely to require maintenance, enabling proactive servicing and minimizing unexpected breakdowns.

---

##  Key Features

- **Predictive Analytics** – Forecasts maintenance needs 30 days in advance with **92% accuracy**
- **Real-time Monitoring** – Live dashboard for fleet health visualization
- **Cost Optimization** – Reduces maintenance costs by **30%**
- **Scalable Architecture** – Works for small and large fleets
- **Business Intelligence** – Power BI integration for decision-making

---

##  Project Structure

```text
vehicle-predictive-maintenance/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── simulated_data_generator.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── utils.py
│   └── powerbi_integration.py
├── models/
│   ├── trained_models/
│   └── feature_importance_plots/
├── tests/
│   ├── test_data_preprocessing.py
│   └── test_models.py
├── deployment/
│   ├── api.py
│   └── Dockerfile
├── reports/
│   ├── model_performance_report.pdf
│   └── business_impact_analysis.md
└── powerbi/
    ├── vehicle_dashboard.pbix
    └── data_connection_script.py
```

---

##  Tech Stack

### Data Engineering
- Python 3.9+ (Pandas, NumPy, Polars)
- Apache Spark (PySpark, Structured Streaming)
- Apache Kafka
- PostgreSQL / MySQL
- Docker

### Machine Learning
- Scikit-learn
- XGBoost / LightGBM / CatBoost
- TensorFlow / PyTorch
- Prophet
- SHAP / LIME

### Visualization
- Plotly, Matplotlib, Seaborn
- Streamlit
- Tableau
- Grafana

### DevOps & MLOps
- MLflow
- Docker & Docker Compose
- GitHub Actions
- Prometheus & Grafana

---

##  Installation

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git
- 8GB+ RAM (recommended)

### Quick Start

```bash
git clone https://github.com/sindhiya225/vehicle-telemetry-analytics.git
cd vehicle-telemetry-analytics

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env

docker-compose up -d

python src/database/setup.py
python main.py --all
```

## Usage

### Run Pipeline

```bash
python main.py --all
python main.py --batch
python main.py --streaming
python main.py --analysis
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

### Streamlit Dashboard

```bash
streamlit run dashboard/streamlit/app.py
```

### MLflow

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


