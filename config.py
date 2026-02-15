import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

class Config:
    INCOME_FACTOR = 100000  # Conversion from lakhs to actual value
    C45_MODEL_PATH = os.path.join(BASE_DIR, 'models/trained_models/classification_model.pkl')
    CLUSTER_MODEL_PATH = os.path.join(BASE_DIR, 'models/trained_models/clustering_model.pkl')
    REGRESSOR_MODEL_PATH = os.path.join(BASE_DIR, 'models/trained_models/regression_model.pkl')
    
    @staticmethod
    def ensure_dirs():
        os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)