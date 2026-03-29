import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import pipeline as hf_pipeline


ROOT_DIR   = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


class CreditLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn * lstm_out).sum(dim=1)
        return self.fc(context).squeeze()


class ModelService:
    def __init__(self) -> None:
        self.xgb_model: Optional[Any] = None
        self.lstm_model: Optional[CreditLSTM] = None
        self.lstm_scaler: Optional[Any] = None
        self.calibrator: Optional[Any] = None
        self.sentiment_pipe: Optional[Any] = None
        self.shap_explainer: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False

    def load_all(self) -> None:
        """Load all models from disk."""
        print("  Loading XGBoost...")
        self.xgb_model = joblib.load(
            MODELS_DIR / "xgboost" / "xgboost_final.pkl")

        print("  Loading SHAP explainer...")
        self.shap_explainer = joblib.load(
            MODELS_DIR / "xgboost" / "shap_explainer.pkl")

        print("  Loading LSTM...")
        checkpoint = torch.load(
            MODELS_DIR / "lstm" / "lstm_checkpoint.pt",
            map_location="cpu")
        self.lstm_model = CreditLSTM(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint.get("hidden_size", 64),
            num_layers=checkpoint.get("num_layers", 2))
        self.lstm_model.load_state_dict(
            checkpoint["model_state_dict"])
        self.lstm_model.eval()
        try:
            self.lstm_scaler = joblib.load(
                MODELS_DIR / "lstm" / "lstm_scaler.pkl")
        except FileNotFoundError:
            print("  Warning: LSTM scaler not found")
            self.lstm_scaler = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lstm_model.to(self.device)

        print("  Loading FinBERT...")
        self.sentiment_pipe = hf_pipeline(
            "text-classification",
            model=str(MODELS_DIR / "finbert" / "finbert_finetuned"),
            tokenizer=str(MODELS_DIR / "finbert" / "finbert_finetuned"),
            device=-1
        )

        print("  Loading calibrator...")
        try:
            self.calibrator = joblib.load(
                MODELS_DIR / "ensemble" / "calibrator.pkl")
        except FileNotFoundError:
            print("  Warning: Calibrator not found, predictions will use XGBoost only")
            self.calibrator = None

        # Load feature names from training data
        processed = ROOT_DIR / "data" / "processed"
        X_test = pd.read_parquet(processed / "X_test.parquet")
        self.feature_names = list(X_test.columns)

        self.is_loaded = True
        print("  All models loaded!")

    def prepare_features(self, request: Any) -> pd.DataFrame:
        """Convert API request to feature vector."""
        try:
            # Ensure feature_names is loaded
            if self.feature_names is None:
                raise RuntimeError("Feature names not loaded. Call load_all() first.")

            # Safe attribute access with defaults
            credit_amount = float(getattr(request, "credit_amount", 0))
            income_total = float(getattr(request, "income_total", 1))
            annuity_amount = float(getattr(request, "annuity_amount", 0))
            goods_price = getattr(request, "goods_price", None)
            family_members = float(getattr(request, "family_members", 1))
            employment_years = getattr(request, "employment_years", None)
            age_years = float(getattr(request, "age_years", 1))
            ext_source_1 = getattr(request, "ext_source_1", None)
            ext_source_2 = getattr(request, "ext_source_2", None)
            ext_source_3 = getattr(request, "ext_source_3", None)
            own_car = bool(getattr(request, "own_car", False))
            own_realty = bool(getattr(request, "own_realty", False))
            car_age = getattr(request, "car_age", None)
            children_count = float(getattr(request, "children_count", 0))

            # Core financial ratios
            debt_to_income = credit_amount / max(income_total, 1)
            annuity_to_income = annuity_amount / max(income_total, 1)
            credit_to_goods = credit_amount / max((goods_price or credit_amount), 1)
            income_per_person = income_total / max(family_members, 1)

            ext_sources = [
                ext_source_1 or np.nan,
                ext_source_2 or np.nan,
                ext_source_3 or np.nan
            ]
            ext_mean: float = float(np.nanmean(ext_sources))
            ext_mean = ext_mean if not np.isnan(ext_mean) else 0.5

            # Build feature dict with defaults (using NaN for unknown features)
            features: Dict[str, float] = {f: np.nan for f in self.feature_names}

            # Fill known features
            feature_map: Dict[str, float] = {
                "AGE_YEARS":            age_years,
                "EMPLOYMENT_YEARS":     employment_years or np.nan,
                "AMT_INCOME_TOTAL":     min(income_total, 472500),
                "AMT_CREDIT":           min(credit_amount, 1854000),
                "AMT_ANNUITY":          annuity_amount,
                "CNT_FAM_MEMBERS":      family_members,
                "CNT_CHILDREN":         children_count,
                "DEBT_TO_INCOME":       debt_to_income,
                "ANNUITY_TO_INCOME":    annuity_to_income,
                "CREDIT_TO_GOODS":      credit_to_goods,
                "INCOME_PER_PERSON":    income_per_person,
                "EXT_SOURCE_1":         ext_source_1 or ext_mean,
                "EXT_SOURCE_2":         ext_source_2 or ext_mean,
                "EXT_SOURCE_3":         ext_source_3 or ext_mean,
                "EXT_SOURCE_MEAN":      ext_mean,
                "FLAG_OWN_CAR":         float(int(own_car)),
                "FLAG_OWN_REALTY":      float(int(own_realty)),
                "OWN_CAR_AGE":          car_age or np.nan,
                "EMPLOYMENT_TO_AGE_RATIO": (
                    (employment_years or 0) / max(age_years, 1)),
                "ANNUITY_TO_CREDIT":    (annuity_amount / max(credit_amount, 1)),
            }

            for k, v in feature_map.items():
                if k in features:
                    features[k] = v

            # Fill remaining NaN with 0.0
            for k in features:
                if pd.isna(features[k]):
                    features[k] = 0.0

            return pd.DataFrame([features])
        except Exception as e:
            print(f"Error preparing features: {e}")
            # Return minimal valid dataframe
            if self.feature_names is None:
                raise RuntimeError("Feature names not loaded")
            return pd.DataFrame([{f: 0.0 for f in self.feature_names}])

    def get_shap_explanation(self, features_df: pd.DataFrame) -> List[Tuple[str, float]]:
        """Get SHAP values for a prediction."""
        try:
            if self.shap_explainer is None or self.feature_names is None:
                return []

            shap_values = self.shap_explainer.shap_values(features_df)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_arr = shap_values

            # Ensure we have the right shape
            if hasattr(shap_arr, "shape") and len(shap_arr.shape) > 1:
                shap_arr = shap_arr[0]

            feature_importance: List[Tuple[str, float]] = list(zip(
                self.feature_names,
                shap_arr
            ))
            feature_importance.sort(key=lambda x: abs(x[1]),
                                      reverse=True)
            return feature_importance[:20]
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return []

    def get_sentiment(self, loan_purpose: str) -> Tuple[str, float]:
        """Run FinBERT sentiment on loan purpose text."""
        if not loan_purpose or not isinstance(loan_purpose, str):
            return "neutral", 0.0

        if self.sentiment_pipe is None:
            return "neutral", 0.0

        try:
            result = self.sentiment_pipe(loan_purpose[:512])[0]
            label: str = str(result.get("label", "neutral")).lower()
            score: float = float(result.get("score", 0.0))
            return label, score
        except Exception as e:
            print(f"Error in sentiment pipeline: {e}")
            return "neutral", 0.0

    def get_lstm_prediction(self, payment_history: Optional[Any] = None) -> float:
        """Get LSTM prediction from payment history.

        Args:
            payment_history: Optional list/array of payment features (SEQ_LENGTH x num_features)

        Returns:
            Default probability score (0.5 if no history available)
        """
        if self.lstm_scaler is None or self.lstm_model is None:
            return 0.5

        # Check if payment history is None or empty
        if payment_history is None:
            return 0.5

        try:
            # Convert to numpy array if it's a list
            if isinstance(payment_history, list):
                payment_history = np.array(payment_history)

            # Check dimensions
            if isinstance(payment_history, np.ndarray):
                if payment_history.size == 0:
                    return 0.5

                # Ensure proper shape (rows, features)
                if len(payment_history.shape) == 1:
                    payment_history = payment_history.reshape(1, -1)

                # Check if we have data
                if payment_history.shape[0] == 0:
                    return 0.5

                # Scale features
                scaled: np.ndarray = self.lstm_scaler.transform(payment_history)
                scaled = np.clip(np.nan_to_num(scaled, nan=0.0), 0, 1)

                # Convert to tensor: (1, seq_len, features)
                tensor: torch.Tensor = torch.FloatTensor(scaled).unsqueeze(0)
                tensor = tensor.to(self.device)

                with torch.no_grad():
                    prob: float = float(torch.sigmoid(self.lstm_model(tensor)).cpu().item())

                return float(np.clip(prob, 0.0, 1.0))
            else:
                return 0.5
        except Exception as e:
            print(f"Warning: LSTM prediction failed ({str(e)}), using default 0.5")
            return 0.5

    def probability_to_score(self, prob: float) -> int:
        """Convert default probability to 300-850 score."""
        score: float = 850 - (prob * (850 - 300))
        return int(np.clip(score, 300, 850))

    def get_risk_band(self, score: int) -> str:
        """Get risk band from credit score."""
        if score >= 750:
            return "Low Risk"
        elif score >= 650:
            return "Medium Risk"
        elif score >= 550:
            return "High Risk"
        else:
            return "Very High Risk"

    def get_recommendation(self, score: int) -> str:
        """Get recommendation from credit score."""
        if score >= 750:
            return "APPROVE"
        elif score >= 650:
            return "APPROVE WITH CONDITIONS"
        elif score >= 550:
            return "MANUAL REVIEW"
        else:
            return "DECLINE"

    def assess(self, request: Any) -> Dict[str, Any]:
        """Full assessment pipeline."""
        start: float = time.time()

        try:
            # Validate required request attributes
            required_attrs = [
                "credit_amount", "income_total", "annuity_amount",
                "family_members", "children_count", "age_years",
                "own_car", "own_realty"
            ]
            for attr in required_attrs:
                if not hasattr(request, attr):
                    raise ValueError(f"Missing required attribute: {attr}")

            # Prepare features
            features_df: pd.DataFrame = self.prepare_features(request)

            # XGBoost prediction
            xgb_proba: float = 0.5
            try:
                if self.xgb_model is not None:
                    xgb_proba = float(
                        self.xgb_model.predict_proba(features_df)[0, 1])
            except Exception as e:
                print(f"Error in XGBoost prediction: {e}")
                xgb_proba = 0.5

            # Sentiment
            sentiment_label: str = "neutral"
            sentiment_conf: float = 0.0
            sentiment_score: float = 0.3
            try:
                loan_purpose: str = str(getattr(request, "loan_purpose", ""))
                sentiment_label, sentiment_conf = self.get_sentiment(loan_purpose)
                sentiment_map: Dict[str, float] = {
                    "negative": 0.7, "neutral": 0.3, "positive": 0.1}
                sentiment_score = sentiment_map.get(sentiment_label, 0.3)
            except Exception as e:
                print(f"Warning: Sentiment analysis failed ({str(e)}), using neutral")
                sentiment_label, sentiment_conf = "neutral", 0.0
                sentiment_score = 0.3

            # LSTM (with payment history if available)
            payment_history: Optional[Any] = getattr(request, "payment_history", None)
            lstm_score: float = self.get_lstm_prediction(payment_history)

            # Calibrated ensemble
            X_ensemble: np.ndarray = np.array([[xgb_proba, lstm_score,
                                      sentiment_score]])
            calibrated_proba: float = xgb_proba
            if self.calibrator is not None:
                try:
                    calibrated_proba = float(
                        self.calibrator.predict_proba(X_ensemble)[0, 1])
                except Exception as e:
                    print(f"Error in calibrator: {e}, using XGBoost prediction")
                    calibrated_proba = xgb_proba
            else:
                calibrated_proba = xgb_proba

            # Credit score
            credit_score: int = self.probability_to_score(calibrated_proba)
            risk_band: str = self.get_risk_band(credit_score)
            recommendation: str = self.get_recommendation(credit_score)

            # SHAP explanation
            top_risk: List[Dict[str, Any]] = []
            top_protective: List[Dict[str, Any]] = []
            try:
                shap_factors: List[Tuple[str, float]] = self.get_shap_explanation(features_df)
                top_risk = [
                    {"feature": f, "importance": float(abs(v)),
                     "direction": "increases_risk"}
                    for f, v in shap_factors if v > 0
                ][:5]
                top_protective = [
                    {"feature": f, "importance": float(abs(v)),
                     "direction": "decreases_risk"}
                    for f, v in shap_factors if v < 0
                ][:5]
            except Exception as e:
                print(f"Warning: SHAP explanation failed ({str(e)}), skipping")
                top_risk = []
                top_protective = []

            inference_time: int = int((time.time() - start) * 1000)

            result: Dict[str, Any] = {
                "credit_score":              credit_score,
                "risk_band":                 risk_band,
                "default_probability":       float(calibrated_proba),
                "recommendation":            recommendation,
                "xgb_default_probability":   float(xgb_proba),
                "lstm_default_probability":  float(lstm_score),
                "sentiment_label":           sentiment_label,
                "sentiment_confidence":      float(sentiment_conf),
                "top_risk_factors":          top_risk,
                "top_protective_factors":    top_protective,
                "model_version":             "1.0.0",
                "inference_time_ms":         inference_time
            }
            return result
        except Exception as e:
            print(f"Critical error in assess: {e}")
            raise