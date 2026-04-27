import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from features import FeatureEngineer, MODEL_FEATURE_COLUMNS
from model import ModelEngine


def prepare_training_data(csv_path="historical_1s.csv", sample_every_seconds=5):
    print(f"Carregando dados de {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Mapeando rodadas de 5 minutos...")
    df["round_start"] = df["timestamp"].dt.floor("5min")
    df["expiry_time"] = df["round_start"] + pd.Timedelta(minutes=5)

    s0_map = df.groupby("round_start")["price"].first()
    df["S0"] = df["round_start"].map(s0_map)

    expiry_map = df.groupby("expiry_time")["price"].last()
    df["outcome_price"] = df["expiry_time"].map(expiry_map)

    df = df.dropna(subset=["outcome_price", "S0"]).copy()
    df["true_label"] = (df["outcome_price"] >= df["S0"]).astype(float)
    df["polymarket_price"] = np.clip(0.5 + ((df["price"] - df["S0"]) / df["S0"]) * 100, 0.01, 0.99)

    print("Extraindo features...")
    df = df.set_index("timestamp")
    fe = FeatureEngineer()
    df_features = fe.build_backtest_features(df, df["S0"], df["expiry_time"]).reset_index()

    if sample_every_seconds > 1:
        within_round_idx = df_features.groupby("expiry_time").cumcount()
        df_features = df_features[within_round_idx % sample_every_seconds == 0].copy()

    df_features["round_weight"] = 1.0 / df_features.groupby("expiry_time")["expiry_time"].transform("count")
    return df_features


def split_by_rounds(df):
    rounds = pd.Series(sorted(df["expiry_time"].unique()))
    train_cut = int(len(rounds) * 0.70)
    val_cut = int(len(rounds) * 0.85)

    train_rounds = set(rounds.iloc[:train_cut])
    val_rounds = set(rounds.iloc[train_cut:val_cut])
    test_rounds = set(rounds.iloc[val_cut:])

    train_df = df[df["expiry_time"].isin(train_rounds)].copy()
    val_df = df[df["expiry_time"].isin(val_rounds)].copy()
    test_df = df[df["expiry_time"].isin(test_rounds)].copy()
    return train_df, val_df, test_df


def print_metrics(label, y_true, probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    print(
        f"{label}: "
        f"Brier={brier_score_loss(y_true, probs):.4f} | "
        f"AUC={roc_auc_score(y_true, probs):.4f} | "
        f"LogLoss={log_loss(y_true, probs):.4f} | "
        f"Acc={(accuracy_score(y_true, probs >= 0.5)):.4f}"
    )


def train_and_save(csv_path="historical_1s.csv", sample_every_seconds=5):
    df = prepare_training_data(csv_path=csv_path, sample_every_seconds=sample_every_seconds)
    train_df, val_df, test_df = split_by_rounds(df)

    print(
        f"Dataset pronto: {len(df)} amostras | "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
        f"rounds={df['expiry_time'].nunique()}"
    )

    model_engine = ModelEngine(input_dim=len(MODEL_FEATURE_COLUMNS), load_existing=False)
    model_engine.feature_columns = list(MODEL_FEATURE_COLUMNS)

    model_engine.fit_sklearn_model(
        train_df[MODEL_FEATURE_COLUMNS].values,
        train_df["true_label"].values,
        sample_weight=train_df["round_weight"].values,
    )

    val_raw = model_engine.predict_raw_proba(val_df[MODEL_FEATURE_COLUMNS].values)
    model_engine.fit_calibrator_from_raw(val_raw, val_df["true_label"].values)

    train_probs = model_engine.predict_proba(torch.tensor(train_df[MODEL_FEATURE_COLUMNS].values, dtype=torch.float32))
    val_probs = model_engine.predict_proba(torch.tensor(val_df[MODEL_FEATURE_COLUMNS].values, dtype=torch.float32))
    test_probs = model_engine.predict_proba(torch.tensor(test_df[MODEL_FEATURE_COLUMNS].values, dtype=torch.float32))

    print_metrics("Train", train_df["true_label"].values, train_probs)
    print_metrics("Val", val_df["true_label"].values, val_probs)
    print_metrics("Test", test_df["true_label"].values, test_probs)

    print("Salvando modelo e calibrador...")
    model_engine.save()
    print("✅ Treinamento concluído com sucesso!")


def main():
    parser = argparse.ArgumentParser(description="Treina o modelo quantitativo do bot BTC 5m.")
    parser.add_argument("--csv-path", default="historical_1s.csv", help="CSV histórico de 1 segundo.")
    parser.add_argument(
        "--sample-every-seconds",
        type=int,
        default=5,
        help="Usa 1 a cada N observações por rodada para reduzir dependência serial.",
    )
    args = parser.parse_args()
    train_and_save(csv_path=args.csv_path, sample_every_seconds=args.sample_every_seconds)


if __name__ == "__main__":
    main()
