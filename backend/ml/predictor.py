"""Load the trained model and score market conditions."""

import pandas as pd
import numpy as np
import pickle
import os
import json
import argparse
from pathlib import Path


class AMQSSPredictor:
    """Load and use trained RandomForest model for market environment classification."""
    
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scaler, and metadata from disk."""
        model_path = os.path.join(self.models_dir, 'model.pkl')
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        meta_path = os.path.join(self.models_dir, 'metadata.json')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
            raise FileNotFoundError(f"Missing model artifacts in {self.models_dir}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_cols = self.metadata['feature_columns']
        print(f"Model loaded from {self.models_dir}")
        print(f"  Features: {self.feature_cols}")
        print(f"  Trained: {self.metadata.get('trained_at', 'unknown')}")
    
    def predict(self, amqss_features):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        X = []
        for col in self.feature_cols:
            if col not in amqss_features:
                raise ValueError(f"Missing feature: {col}")
            X.append(amqss_features[col])
        
        X = np.array([X], dtype=float)
        
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)[0][1]
        
        return float(proba)
    
    def predict_batch(self, amqss_features_list):
        X = []
        for features in amqss_features_list:
            row = [features.get(col, 0) for col in self.feature_cols]
            X.append(row)
        
        X = np.array(X, dtype=float)
        X_scaled = self.scaler.transform(X)
        probas = self.model.predict_proba(X_scaled)[:, 1]
        
        return probas


def load_predictor(models_dir):
    """Convenience function to load predictor."""
    return AMQSSPredictor(models_dir)


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Load trained AMQSS predictor and run sample prediction')
    parser.add_argument(
        '--models-dir',
        default=os.getenv('AMQSS_MODELS_DIR', str(project_root / 'models')),
        help='Directory containing model artifacts'
    )
    args = parser.parse_args()

    models_dir = str(Path(args.models_dir).expanduser())
    
    try:
        predictor = load_predictor(models_dir)
        
        example_features = {
            'volatility_regime': 5.0,
            'true_liquidity': 7.0,
            'structure': 8.0,
            'momentum': 6.0,
            'volatility_level': 4.0,
            'current_atr': 0.001
        }
        
        prob = predictor.predict(example_features)
        print(f"\nExample prediction: {prob:.4f}")
        print(f"Market environment quality: {'HIGH' if prob > 0.5 else 'LOW'}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
