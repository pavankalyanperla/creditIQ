import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


class ModelServiceLite:
    """
    Lightweight model service for deployment.
    Uses only XGBoost + SHAP (no FinBERT/LSTM).
    """

    def __init__(self) -> None:
        self.xgb_model = None
        self.shap_explainer = None
        self.calibrator = None
        self.feature_names = None
        self.is_loaded = False

    def load_all(self) -> None:
        print("  Loading XGBoost...")
        self.xgb_model = joblib.load(MODELS_DIR / "xgboost" / "xgboost_final.pkl")

        print("  Loading SHAP explainer...")
        try:
            self.shap_explainer = joblib.load(
                MODELS_DIR / "xgboost" / "shap_explainer.pkl"
            )
        except FileNotFoundError:
            print("  Warning: SHAP explainer not found")

        print("  Loading calibrator...")
        try:
            self.calibrator = joblib.load(MODELS_DIR / "ensemble" / "calibrator.pkl")
        except FileNotFoundError:
            print("  Warning: Calibrator not found")

        # Load feature names
        processed = ROOT_DIR / "data" / "processed"
        X_test = pd.read_parquet(processed / "X_test.parquet")
        self.feature_names = list(X_test.columns)

        self.is_loaded = True
        print("  Lite models loaded!")

    def prepare_features(self, request: Any) -> pd.DataFrame:
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

        ext_sources = [
            ext_source_1 or np.nan,
            ext_source_2 or np.nan,
            ext_source_3 or np.nan,
        ]
        ext_mean = float(np.nanmean(ext_sources))
        ext_mean = ext_mean if not np.isnan(ext_mean) else 0.5

        features = {f: 0.0 for f in (self.feature_names or [])}

        feature_map = {
            "AGE_YEARS": age_years,
            "EMPLOYMENT_YEARS": employment_years or 0,
            "AMT_INCOME_TOTAL": min(income_total, 472500),
            "AMT_CREDIT": min(credit_amount, 1854000),
            "AMT_ANNUITY": annuity_amount,
            "CNT_FAM_MEMBERS": family_members,
            "CNT_CHILDREN": children_count,
            "DEBT_TO_INCOME": credit_amount / max(income_total, 1),
            "ANNUITY_TO_INCOME": annuity_amount / max(income_total, 1),
            "CREDIT_TO_GOODS": credit_amount / max((goods_price or credit_amount), 1),
            "INCOME_PER_PERSON": income_total / max(family_members, 1),
            "EXT_SOURCE_1": ext_source_1 or ext_mean,
            "EXT_SOURCE_2": ext_source_2 or ext_mean,
            "EXT_SOURCE_3": ext_source_3 or ext_mean,
            "EXT_SOURCE_MEAN": ext_mean,
            "FLAG_OWN_CAR": float(int(own_car)),
            "FLAG_OWN_REALTY": float(int(own_realty)),
            "OWN_CAR_AGE": car_age or 0,
            "EMPLOYMENT_TO_AGE_RATIO": (employment_years or 0) / max(age_years, 1),
            "ANNUITY_TO_CREDIT": annuity_amount / max(credit_amount, 1),
        }

        for k, v in feature_map.items():
            if k in features:
                features[k] = v

        return pd.DataFrame([features])

    def get_shap_explanation(
        self, features_df: pd.DataFrame
    ) -> List[Tuple[str, float]]:
        if self.shap_explainer is None or self.feature_names is None:
            return []
        try:
            shap_values = self.shap_explainer.shap_values(features_df)
            if isinstance(shap_values, list):
                shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_arr = shap_values
            if hasattr(shap_arr, "shape") and len(shap_arr.shape) > 1:
                shap_arr = shap_arr[0]
            importance = list(zip(self.feature_names, shap_arr))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            return importance[:20]
        except Exception as e:
            print(f"SHAP error: {e}")
            return []

    def probability_to_score(self, prob: float) -> int:
        score = 850 - (prob * (850 - 300))
        return int(np.clip(score, 300, 850))

    def get_risk_band(self, score: int) -> str:
        if score >= 750:
            return "Low Risk"
        elif score >= 650:
            return "Medium Risk"
        elif score >= 550:
            return "High Risk"
        else:
            return "Very High Risk"

    def get_recommendation(self, score: int) -> str:
        if score >= 750:
            return "APPROVE"
        elif score >= 650:
            return "APPROVE WITH CONDITIONS"
        elif score >= 550:
            return "MANUAL REVIEW"
        else:
            return "DECLINE"

    def assess(self, request: Any) -> Dict[str, Any]:
        start = time.time()

        features_df = self.prepare_features(request)

        # XGBoost prediction
        xgb_proba = 0.5
        if self.xgb_model is not None:
            xgb_proba = float(
                self.xgb_model.predict_proba(features_df)[0, 1]
            )  # type: ignore

        # Calibrate if available
        calibrated_proba = xgb_proba
        if self.calibrator is not None:
            try:
                X_ens = np.array([[xgb_proba, 0.5, 0.3]])
                calibrated_proba = float(
                    self.calibrator.predict_proba(X_ens)[0, 1]
                )  # type: ignore
            except Exception:
                calibrated_proba = xgb_proba

        credit_score = self.probability_to_score(calibrated_proba)
        risk_band = self.get_risk_band(credit_score)
        recommendation = self.get_recommendation(credit_score)

        # SHAP
        shap_factors = self.get_shap_explanation(features_df)
        top_risk = [
            {"feature": f, "importance": float(abs(v)), "direction": "increases_risk"}
            for f, v in shap_factors
            if v > 0
        ][:5]
        top_protective = [
            {"feature": f, "importance": float(abs(v)), "direction": "decreases_risk"}
            for f, v in shap_factors
            if v < 0
        ][:5]

        return {
            "credit_score": credit_score,
            "risk_band": risk_band,
            "default_probability": float(calibrated_proba),
            "recommendation": recommendation,
            "xgb_default_probability": float(xgb_proba),
            "lstm_default_probability": 0.5,
            "sentiment_label": "neutral",
            "sentiment_confidence": 0.0,
            "top_risk_factors": top_risk,
            "top_protective_factors": top_protective,
            "model_version": "1.0.0-lite",
            "inference_time_ms": int((time.time() - start) * 1000),
        }
