"""Serializable, calibrated clinical model used by the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CalibratedXGBoostModel:
    """XGBoost estimator plus imputation and an OOF-fitted sigmoid calibrator.

    Keeping this class in an importable module makes the joblib artifact portable.
    The delegated XGBoost methods preserve the dashboard's TreeSHAP support.
    """

    def __init__(self, imputer, estimator, calibrator, feature_names, threshold, metadata=None):
        self.imputer = imputer
        self.estimator = estimator
        self.calibrator = calibrator
        self.feature_names = list(feature_names)
        self.threshold_ = float(threshold)
        self.metadata_ = metadata or {}
        self.classes_ = np.asarray([0, 1])
        self.n_features_in_ = len(self.feature_names)

    @staticmethod
    def _logit(probability):
        probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(probability / (1 - probability)).reshape(-1, 1)

    def predict_proba(self, values):
        values = self._named_frame(values)
        transformed = self.imputer.transform(values)
        raw = self.estimator.predict_proba(transformed)[:, 1]
        calibrated = self.calibrator.predict_proba(self._logit(raw))[:, 1]
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, values):
        return (self.predict_proba(values)[:, 1] >= self.threshold_).astype(int)

    def get_booster(self):
        return self.estimator.get_booster()

    def shap_contributions(self, values):
        """Return exact TreeSHAP values after the training-time imputation."""
        import xgboost as xgb

        transformed = self.imputer.transform(self._named_frame(values))
        return self.estimator.get_booster().predict(
            xgb.DMatrix(transformed), pred_contribs=True
        )

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    def _named_frame(self, values):
        if isinstance(values, pd.DataFrame):
            return values.loc[:, self.feature_names]
        return pd.DataFrame(values, columns=self.feature_names)
