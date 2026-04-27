import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import IsotonicRegression
import numpy as np
import os
import joblib

class ProfitAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, p_mkt):
        """
        Custom Profit-Aware Loss: Brier Score weighted by distance to market price.
        Penaliza erros maiores quando o mercado está precificando de forma mais extrema 
        em relação ao resultado verdadeiro.
        """
        brier_score = (y_pred - y_true) ** 2
        # Peso baseado no quão "errado" o mercado está. 
        # Se mercado = 0.9 e true = 0, weight = 0.9. (oportunidade grande)
        weight = torch.abs(p_mkt - y_true) + 0.1 # +0.1 para evitar zero weight
        loss = brier_score * weight
        return loss.mean()

class QuantMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ModelEngine:
    def __init__(self, input_dim, lr=1e-3, weight_decay=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QuantMLP(input_dim).to(self.device)
        self.criterion = ProfitAwareLoss()
        
        # Otimizador AdamW
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Calibrador
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        
        # Carrega pesos se existirem
        if os.path.exists("model_weights.pth"):
            self.model.load_state_dict(torch.load("model_weights.pth", map_location=self.device, weights_only=True))
            print("🧠 Pesos da Rede Neural carregados com sucesso!")
            
        if os.path.exists("calibrator.pkl"):
            self.calibrator = joblib.load("calibrator.pkl")
            print("🎯 Calibrador Isotonic Regression carregado com sucesso!")
        
    def train_epoch(self, X, y, p_mkt, batch_size=256):
        self.model.train()
        
        # CosineAnnealingWarmRestarts para LR
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        permutation = torch.randperm(X.size()[0])
        
        total_loss = 0
        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y, batch_p = X[indices], y[indices], p_mkt[indices]
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_p = batch_p.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(batch_x)
            loss = self.criterion(preds, batch_y, batch_p)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        return total_loss / (X.size()[0] / batch_size)
        
    def fit_calibrator(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_val.to(self.device)).cpu().numpy()
        self.calibrator.fit(preds, y_val.numpy())

    def predict_raw_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device)).cpu().numpy()

    def calibrate_raw_proba(self, raw_preds):
        raw_preds = np.asarray(raw_preds, dtype=np.float64)
        if hasattr(self.calibrator, 'X_min_'):
            return self.calibrator.transform(raw_preds)
        return raw_preds

    def fit_calibrator_from_raw(self, raw_preds, y_true):
        raw_preds = np.asarray(raw_preds, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_preds, y_true)
        
    def predict_proba(self, X):
        raw_preds = self.predict_raw_proba(X)
        return self.calibrate_raw_proba(raw_preds)
