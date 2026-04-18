import numpy as np
import torch
import torch.nn as nn
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
SEQ = 14

# ============================================================
# 特征工程（与 test.py 完全一致）
# ============================================================
def _build_features(series_norm):
    returns = np.concatenate([[0], np.diff(series_norm)])
    ma3 = np.convolve(series_norm, np.ones(3)/3, mode='full')[:len(series_norm)]
    ma3[:2] = series_norm[:2]
    return np.stack([series_norm, returns, series_norm - ma3], axis=1).astype(np.float32)

# ============================================================
# 模型（与 test.py 完全一致）
# ============================================================
class LightAttentionLSTM(nn.Module):
    def __init__(self, n_feat=3, hidden=40):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=False)
        self.attn_w = nn.Linear(hidden, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 40),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(40, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn_w(out), dim=0)
        return self.fc((w * out).sum(dim=0))

# ============================================================
# 加载模型（一次性，模块级）
# ============================================================
_model = LightAttentionLSTM()
_model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, 'results', 'mymodel.pt'), map_location='cpu'))
_model.eval()

# ============================================================
# predict：每只股票用自身序列单独标准化再逆变换
# 关键：与训练时逐股票标准化保持一致，确保 MAE 不因价格尺度放大
# ============================================================
def predict(data):
    """
    输入: data, shape (n_stocks, T) 或 (T,)，T >= 14
    输出: numpy array, shape (n_stocks, 1)，原始收盘价
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    results = []
    for s in range(data.shape[0]):
        series = data[s].astype(np.float64)

        # 逐股票标准化
        m   = series.mean()
        std = series.std() + 1e-8
        series_norm = (series - m) / std

        feat = _build_features(series_norm.astype(np.float32))[-SEQ:]
        x = torch.tensor(feat).unsqueeze(1)  # (14, 1, 3)

        with torch.no_grad():
            pred_norm = _model(x).item()

        # 逆变换回原始价格
        results.append(pred_norm * std + m)

    return np.array(results, dtype=np.float64).reshape(-1, 1)
