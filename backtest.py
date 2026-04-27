import pandas as pd
import numpy as np
import torch
from model import ModelEngine
from features import FeatureEngineer
from config import TRAIN_WINDOW_HOURS, TEST_WINDOW_HOURS, SLIPPAGE_CENTS_PER_500, EXECUTION_DELAY_SEC

class WalkForwardBacktester:
    def __init__(self, data_df):
        """
        data_df: Dados históricos contendo: 
        timestamp, price, bid_vol, ask_vol, polymarket_price, S0, expiry_time
        """
        self.data = data_df.copy()
        self.data.set_index('timestamp', inplace=True)
        self.fe = FeatureEngineer()
        
    def run(self):
        print("Iniciando Walk-Forward Backtest...")
        
        # Gera todas as features de forma vetorizada
        df_features = self.fe.build_backtest_features(
            self.data, 
            self.data['S0'], 
            self.data['expiry_time']
        )
        
        # Simula label true (1 se preço_final > S0, 0 caso contrário)
        # Vamos assumir que os dados já contém 'true_label' ou nós os calculamos
        
        feature_cols = [c for c in df_features.columns if c not in ['price', 'bid_vol', 'ask_vol', 'polymarket_price', 'S0', 'expiry_time', 'true_label']]
        
        start_time = df_features.index.min()
        end_time = df_features.index.max()
        
        current_time = start_time + pd.Timedelta(hours=TRAIN_WINDOW_HOURS)
        
        results = []
        
        while current_time + pd.Timedelta(hours=TEST_WINDOW_HOURS) <= end_time:
            train_start = current_time - pd.Timedelta(hours=TRAIN_WINDOW_HOURS)
            test_end = current_time + pd.Timedelta(hours=TEST_WINDOW_HOURS)
            
            train_df = df_features[(df_features.index >= train_start) & (df_features.index < current_time)]
            test_df = df_features[(df_features.index >= current_time) & (df_features.index < test_end)]
            
            if len(train_df) < 1000 or len(test_df) == 0:
                current_time += pd.Timedelta(hours=TEST_WINDOW_HOURS)
                continue
                
            model = ModelEngine(input_dim=len(feature_cols))
            
            X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
            y_train = torch.tensor(train_df['true_label'].values, dtype=torch.float32)
            p_mkt_train = torch.tensor(train_df['implied_prob'].values, dtype=torch.float32)
            
            # Treino
            model.train_epoch(X_train, y_train, p_mkt_train)
            model.fit_calibrator(X_train, y_train) # Na prática devia ser hold-out set
            
            # Teste
            X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
            test_df['P_model'] = model.predict_proba(X_test)
            
            # Simulação de Slippage/Delay e Execução
            test_df['delayed_implied_prob'] = test_df['implied_prob'].shift(-EXECUTION_DELAY_SEC)
            test_df.dropna(inplace=True)
            
            # Condição: Valor Esperado (EV) > 0
            # EV = (P_model * Payout) - Custo
            # Payout = 1 se ganha. Custo = P_mkt + slippage
            test_df['slippage'] = (500 / 500) * SLIPPAGE_CENTS_PER_500 # Simplificando para aposta de $500
            test_df['cost'] = test_df['delayed_implied_prob'] + test_df['slippage']
            test_df['EV'] = test_df['P_model'] * 1.0 - test_df['cost']
            
            # Trades onde EV > Margin
            margin = 0.05
            trades = test_df[test_df['P_model'] > test_df['cost'] + margin]
            
            # Calcular PnL
            trades['PnL'] = np.where(trades['true_label'] == 1, 1.0 - trades['cost'], -trades['cost'])
            results.append(trades)
            
            current_time += pd.Timedelta(hours=TEST_WINDOW_HOURS)
            
        if results:
            all_trades = pd.concat(results)
            print(f"Total Trades: {len(all_trades)}")
            print(f"Total PnL: {all_trades['PnL'].sum():.2f}")
            print(f"Win Rate: {(all_trades['PnL'] > 0).mean():.2%}")
            return all_trades
        return None
