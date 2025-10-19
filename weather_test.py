import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import argparse
import json
import os
from s3io import read_csv

# =========================
#      LOAD MODEL UTILS
# =========================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_outputs, channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else channels[i-1]
            out_channels = channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size-1) * dilation, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(channels[-1], num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        o = self.network(x)
        o = self.out(o[:, :, -1])
        return o

# =========================
#      TEST FUNCTION
# =========================
def prepare_sequences_for_checkpoint(df: pd.DataFrame, checkpoint: dict):
    # Build targets from names like "temperature_t+10" -> base="temperature", horizon=10
    features = checkpoint["features"]
    targets_names = checkpoint["targets"]
    # Parse horizon from any target name (they are consistent)
    horizons = []
    bases = []
    for t in targets_names:
        if "_t+" in t:
            b, h = t.rsplit("_t+", 1)
            bases.append(b)
            horizons.append(int(h))
        else:
            # fallback assume same name as base and checkpoint has 'horizon'
            bases.append(t)
            horizons.append(int(checkpoint.get("horizon", 10)))
    # Use the first horizon if all equal, otherwise per-column shifting accordingly
    # Respect fixed 'timestamp' and assume data already ordered; no sorting
    ts_col = checkpoint.get("timestamp_col", "timestamp")
    df_feat = df[features].copy()
    for base, h, tname in zip(bases, horizons, targets_names):
        df_feat[tname] = df[base].shift(-h)
    df_feat = df_feat.dropna().reset_index(drop=True)

    # Scale features
    if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint['scaler_mean'])
        scaler.scale_ = np.array(checkpoint['scaler_scale'])
        # sklearn expects var_ sometimes; provide for completeness
        scaler.var_ = scaler.scale_ ** 2
        X = (df_feat[features].values - scaler.mean_) / scaler.scale_
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_feat[features].values)
    y = df_feat[targets_names].values

    # Sequence length
    seq_len = int(checkpoint.get("seq_len", 30))
    def create_sequences(X_, y_, seq_len_):
        Xs, ys = [], []
        for i in range(len(X_) - seq_len_):
            Xs.append(X_[i:i+seq_len_])
            ys.append(y_[i+seq_len_-1])
        return np.array(Xs), np.array(ys)
    X_seq, y_seq = create_sequences(X, y, seq_len)
    return X_seq, y_seq

def evaluate_model(model_path, df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    X_seq, y_seq = prepare_sequences_for_checkpoint(df, checkpoint)
    model = TCN(
        num_inputs=len(checkpoint["features"]),
        num_outputs=len(checkpoint["targets"]),
        channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loss_fn = torch.nn.MSELoss()
    X = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y = torch.tensor(y_seq, dtype=torch.float32).to(device)
    if X.dim() == 2:
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)
    elif X.dim() == 1:
        # If only one sequence and one feature, expand to (1, 1, features)
        X = X.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
    start_time = time.time()
    with torch.no_grad():
        pred = model(X)
        mse = loss_fn(pred, y).item()
    latency = time.time() - start_time
    return mse, latency

# =========================
#         MAIN
# =========================
def main(args):
    # Load test data
    df = read_csv(args.data_path)
    # df = df.sort_values(by="date").reset_index(drop=True)
    # Evaluate both models (each with its own metadata-driven preprocessing)
    mse1, latency1 = evaluate_model(args.model1_path, df)
    mse2, latency2 = evaluate_model(args.model2_path, df)
    print(f"Model 1: {args.model1_path}\n  MSE: {mse1:.6f}\n  Latency: {latency1:.4f}s")
    print(f"Model 2: {args.model2_path}\n  MSE: {mse2:.6f}\n  Latency: {latency2:.4f}s")
    # Select best (by MSE primarily; tie-breaker by lower latency)
    if (mse1 < mse2) or (np.isclose(mse1, mse2) and latency1 <= latency2):
        best_path = args.model1_path
    else:
        best_path = args.model2_path
    print(f"Best model: {best_path}")

    # Write machine-readable results and copy best model
    out_dir = args.out_dir or os.path.dirname(best_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "model1": {"path": args.model1_path, "mse": mse1, "latency_sec": latency1},
        "model2": {"path": args.model2_path, "mse": mse2, "latency_sec": latency2},
        "best_model_path": best_path,
        "criteria": "min_mse_then_latency"
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # Copy best to a canonical filename for Jenkins
    try:
        import shutil
        shutil.copy2(best_path, os.path.join(out_dir, "best_model.pth"))
    except Exception as e:
        print(f"Warning: could not copy best model to {out_dir}/best_model.pth: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and compare two weather models.")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the weather_dataset.csv file.')
    parser.add_argument('--model1-path', type=str, required=True, help='Path to first model.pth file.')
    parser.add_argument('--model2-path', type=str, required=True, help='Path to second model.pth file.')
    parser.add_argument('--out-dir', type=str, default='model', help='Directory to write results and best_model.pth')
    args = parser.parse_args()
    main(args)
