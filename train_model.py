import pandas as pd
import numpy as np
import torch
import joblib
from datetime import timedelta
from model import ModelEngine
from features import FeatureEngineer

def simulate_polymarket_price(df):
    """Mock the polymarket price based on distance from S0, same as forward test"""
    dist = (df['price'] - df['S0']) / df['S0']
    prob = 0.5 + (dist * 100)
    return np.clip(prob, 0.01, 0.99)

def prepare_training_data(csv_path="historical_1s.csv"):
    print(f"Carregando dados de {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print("Mapeando Rodadas de 5 Minutos (S0, Expiry, Outcomes)...")
    # Arredonda timestamp para baixo no múltiplo de 5 min mais próximo
    df['round_start'] = df['timestamp'].dt.floor('5min')
    df['expiry_time'] = df['round_start'] + pd.Timedelta(minutes=5)
    
    # Pega o preço no exato início da rodada (S0)
    s0_map = df.groupby('round_start')['price'].first()
    df['S0'] = df['round_start'].map(s0_map)
    
    # Pega o preço no fim da rodada para calcular o resultado verdadeiro
    expiry_map = df.groupby('expiry_time')['price'].last()
    df['outcome_price'] = df['expiry_time'].map(expiry_map)
    
    # Remove as rodadas que ainda não terminaram (outcome_price = NaN)
    df = df.dropna(subset=['outcome_price', 'S0'])
    
    # True Label
    df['true_label'] = (df['outcome_price'] >= df['S0']).astype(float)
    
    # Polymarket Price Mock
    df['polymarket_price'] = simulate_polymarket_price(df)
    
    print("Extraindo Features (Isso pode demorar alguns segundos)...")
    df.set_index('timestamp', inplace=True)
    fe = FeatureEngineer()
    df_features = fe.build_backtest_features(df, df['S0'], df['expiry_time'])
    
    return df_features

def train_and_save():
    df = prepare_training_data()
    
    feature_cols = [
        'time_to_expiry', 'distance_to_strike', 'obi', 'implied_prob',
        'log_return_10s', 'volatility_10s', 'log_return_30s', 'volatility_30s',
        'log_return_60s', 'volatility_60s', 'log_return_300s', 'volatility_300s'
    ]
    
    # Garante que todas as features existem (12 features)
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df['true_label'].values, dtype=torch.float32)
    p_mkt = torch.tensor(df['implied_prob'].values, dtype=torch.float32)
    
    print(f"Dataset pronto: {X.shape[0]} amostras. Treinando modelo...")
    
    # Split train/val para calibração
    train_size = int(0.8 * len(X))
    X_train, y_train, p_mkt_train = X[:train_size], y[:train_size], p_mkt[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    model_engine = ModelEngine(input_dim=len(feature_cols))
    
    epochs = 20
    for epoch in range(epochs):
        loss = model_engine.train_epoch(X_train, y_train, p_mkt_train, batch_size=1024)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
        
    print("Ajustando o Calibrador IsotonicRegression...")
    model_engine.fit_calibrator(X_val, y_val)
    
    print("Salvando Pesos e Calibrador...")
    torch.save(model_engine.model.state_dict(), "model_weights.pth")
    joblib.dump(model_engine.calibrator, "calibrator.pkl")
    print("✅ Treinamento concluído com sucesso! Pesos salvos em 'model_weights.pth'.")

if __name__ == "__main__":
    train_and_save()
