import numpy as np
import torch
import torch.nn as nn
import math
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
os.makedirs('results', exist_ok=True)

# ============================================================
# 数据加载
# ============================================================
data = np.load('train_data.npy')
if data.ndim == 1:
    data = data.reshape(1, -1)

n_stocks, T = data.shape
SEQ = 14
print(f"数据: {n_stocks} 只股票, {T} 步")

# ============================================================
# 特征工程：逐股票标准化（与 predict 保持一致）
# ============================================================
def build_features(series_norm):
    returns = np.concatenate([[0], np.diff(series_norm)])
    ma3 = np.convolve(series_norm, np.ones(3)/3, mode='full')[:len(series_norm)]
    ma3[:2] = series_norm[:2]
    return np.stack([series_norm, returns, series_norm - ma3], axis=1).astype(np.float32)

# ============================================================
# 构造样本
# ============================================================
X, Y = [], []
for s in range(n_stocks):
    series = data[s]
    m, std = series.mean(), series.std() + 1e-8
    series_norm = (series - m) / std
    feat = build_features(series_norm)
    for i in range(SEQ, len(series_norm)):
        X.append(feat[i-SEQ:i])
        Y.append(series_norm[i])

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

split = int(len(X) * 0.9)
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]
print(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}")

Xtr = torch.tensor(X_train).permute(1,0,2).to(device)  # (14, N, 3)
Ytr = torch.tensor(Y_train).to(device)
Xv  = torch.tensor(X_val).permute(1,0,2).to(device)
Yv  = torch.tensor(Y_val).to(device)

# 训练集均值/std，用于将 norm 空间 MAE 换算成真实价格 MAE（监控用）
train_std = float(data[0].std())

# ============================================================
# 模型：hidden=40，适当扩容同时保持正则
# 样本/参数 = 1786 / 8130 ≈ 0.22，仍在安全范围
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

model = LightAttentionLSTM().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_params:,}  样本/参数={len(X_train)/n_params:.3f}")

# ============================================================
# 评分函数（复现平台公式，用于保存最优模型）
# 得分 = max((1-err_mape²)×40,0) + max((1-err_mae²)×60,0)
# err_mape = MAPE×4,  err_mae = sqrt(MAE+1)/3
# ============================================================
def platform_score(mae_real, mape_real):
    err_mae  = math.sqrt(mae_real + 1) / 3
    err_mape = mape_real * 4
    return max((1 - err_mae**2)  * 60, 0) + \
           max((1 - err_mape**2) * 40, 0)

# ============================================================
# 损失函数：MAE 主导（当前阶段降 MAE 收益远大于降 MAPE）
#
# 分析（基于当前测试集结果 MAE=0.92, MAPE=3.27%）：
#   MAE 0.92->0.1 可得 +5.5 分
#   MAPE 3.27%->0% 只得  +0.7 分
#
# 在标准化空间同时优化两者：
#   MAE_norm ≈ MAE_real / std_stock（直接对应原始误差）
#   MAPE_norm 用偏移分母防止除零（标准化后均值~0）
# ============================================================
def loss_fn(pred, target):
    mae  = (pred - target).abs().mean()
    # 分母加偏移：标准化空间中 target 均值~0，直接除会不稳定
    mape = ((pred - target).abs() / (target.abs() + 0.3)).mean()
    return 0.75 * mae + 0.25 * mape

# ============================================================
# 优化器：CosineAnnealingWarmRestarts 多周期重启
# 比单次退火更容易跳出局部最优
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.5e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=200, T_mult=2, eta_min=1e-5
)
# 总 epoch = T_0 + T_0*T_mult + T_0*T_mult² = 200+400+800 = 1400

EPOCHS = 1400
best_score = -1e9
NOISE_STD = 0.005  # 训练时加微小噪声，提升对陌生股票的泛化

print(f"\n开始训练 {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()

    # 加入轻微高斯噪声：模拟测试集不同股票的特征分布偏差
    noise = torch.randn_like(Xtr) * NOISE_STD
    loss = loss_fn(model(Xtr + noise), Ytr)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # ---- 验证 ----
    model.eval()
    with torch.no_grad():
        pred_v = model(Xv)
        mae_norm  = (pred_v - Yv).abs().mean().item()
        mape_norm = ((pred_v - Yv).abs() / (Yv.abs() + 0.3)).mean().item()

        # 换算成真实空间（近似，仅用于监控）
        mae_real  = mae_norm  * train_std
        mape_real = mape_norm  # MAPE 无量纲，近似相同

    est = platform_score(mae_real, mape_real)

    if est > best_score:
        best_score = est
        torch.save(model.state_dict(), 'results/mymodel.pt')
        tag = " ← saved"
    else:
        tag = ""

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | MAE≈{mae_real:.4f}元 | MAPE≈{mape_real*100:.2f}% "
              f"| 预估={est:.1f}分{tag}")

print(f"\n训练完成 ✓  Best 预估得分={best_score:.1f}")
