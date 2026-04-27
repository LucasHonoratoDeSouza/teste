import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from features import MODEL_FEATURE_COLUMNS
from model import ModelEngine
from train_model import prepare_training_data


class WalkForwardBacktester:
    def __init__(self, csv_path="historical_1s.csv", sample_every_seconds=5, train_rounds=120, test_rounds=24):
        self.data = prepare_training_data(csv_path=csv_path, sample_every_seconds=sample_every_seconds)
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds

    def run(self):
        print("Iniciando walk-forward backtest quantitativo...")
        rounds = sorted(self.data["expiry_time"].unique())
        if len(rounds) < self.train_rounds + self.test_rounds:
            raise ValueError("Histórico insuficiente para o walk-forward configurado.")

        all_predictions = []

        start_idx = self.train_rounds
        while start_idx + self.test_rounds <= len(rounds):
            train_round_list = rounds[start_idx - self.train_rounds:start_idx]
            train_slice = set(train_round_list)
            test_slice = set(rounds[start_idx:start_idx + self.test_rounds])

            train_df = self.data[self.data["expiry_time"].isin(train_slice)].copy()
            split_point = int(len(train_round_list) * 0.8)
            calib_rounds = set(train_round_list[split_point:])
            fit_rounds = train_slice - calib_rounds
            fit_df = train_df[train_df["expiry_time"].isin(fit_rounds)].copy()
            calib_df = train_df[train_df["expiry_time"].isin(calib_rounds)].copy()
            test_df = self.data[self.data["expiry_time"].isin(test_slice)].copy()

            model_engine = ModelEngine(input_dim=len(MODEL_FEATURE_COLUMNS), load_existing=False)
            model_engine.feature_columns = list(MODEL_FEATURE_COLUMNS)
            model_engine.fit_sklearn_model(
                fit_df[MODEL_FEATURE_COLUMNS].values,
                fit_df["true_label"].values,
                sample_weight=fit_df["round_weight"].values,
            )

            calib_raw = model_engine.predict_raw_proba(calib_df[MODEL_FEATURE_COLUMNS].values)
            model_engine.fit_calibrator_from_raw(calib_raw, calib_df["true_label"].values)

            test_df["P_model"] = model_engine.predict_proba(test_df[MODEL_FEATURE_COLUMNS].values)
            all_predictions.append(test_df[["expiry_time", "true_label", "P_model"]])

            start_idx += self.test_rounds

        predictions = pd.concat(all_predictions, ignore_index=True)
        y_true = predictions["true_label"].values
        probs = np.clip(predictions["P_model"].values, 1e-6, 1 - 1e-6)

        print(f"Amostras avaliadas: {len(predictions)}")
        print(f"Brier Score : {brier_score_loss(y_true, probs):.4f}")
        print(f"AUC         : {roc_auc_score(y_true, probs):.4f}")
        print(f"Log Loss    : {log_loss(y_true, probs):.4f}")
        print(f"Accuracy    : {accuracy_score(y_true, probs >= 0.5):.4f}")
        return predictions
