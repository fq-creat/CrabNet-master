# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device


# ===============================
# 全局设置
# ===============================
compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ===============================
# Min-Max 归一化工具
# ===============================
def minmax_scale(y, y_min, y_max):
    return (y - y_min) / (y_max - y_min)


def minmax_inverse(y_norm, y_min, y_max):
    return y_norm * (y_max - y_min) + y_min


# ===============================
# 模型训练
# ===============================
def get_model(data_dir, mat_prop, classification=False,
              transfer=None, verbose=True):

    model = Model(
        CrabNet(compute_device=compute_device).to(compute_device),
        model_name=f'{mat_prop}',
        verbose=verbose
    )

    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    if classification:
        model.classification = True

    train_data = f'{data_dir}/{mat_prop}/train.csv'

    # ====== batch size 计算 ======
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2 ** round(np.log2(data_size) - 4)
    batch_size = min(max(batch_size, 2**7), 2**12)

    # ✅ 只加载训练集
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size}')

    model.fit(epochs=500, losscurve=False)
    model.save_network()
    return model


# ===============================
# 统一导出 Excel（反归一化）
# ===============================
def save_prediction_excel(output, mat_prop, split_name):
    y_true_norm, y_pred_norm, formulae, uncertainty = output

    # 读取训练集归一化参数
    norm_param = pd.read_csv(f"normalization/{mat_prop}_minmax.csv")
    y_min = norm_param["y_min"].iloc[0]
    y_max = norm_param["y_max"].iloc[0]

    # 反归一化
    y_true = minmax_inverse(y_true_norm, y_min, y_max)
    y_pred = minmax_inverse(y_pred_norm, y_min, y_max)

    df = pd.DataFrame({
        "formula": formulae,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    df["dataset"] = split_name

    os.makedirs("model_predictions", exist_ok=True)
    save_path = f"model_predictions/{mat_prop}_{split_name}_predictions.xlsx"
    df.to_excel(save_path, index=False)

    print(f"✅ Prediction table saved to: {save_path}")


# ===============================
# 模型加载
# ===============================
def load_model(data_dir, mat_prop, classification, file_name, verbose=True):
    model = Model(
        CrabNet(compute_device=compute_device).to(compute_device),
        model_name=f'{mat_prop}',
        verbose=verbose
    )
    model.load_network(f'{mat_prop}.pth')

    if classification:
        model.classification = True

    data = f'{data_dir}/{mat_prop}/{file_name}'
    model.load_data(data, batch_size=2**9, train=False)
    return model


# ===============================
# 评估与保存（反归一化后算指标）
# ===============================
def save_results(data_dir, mat_prop, classification, file_name, verbose=True):
    model = load_model(data_dir, mat_prop, classification, file_name, verbose)
    output = model.predict(model.data_loader)

    y_true_norm, y_pred_norm = output[0], output[1]
    split_name = file_name.replace(".csv", "")

    # 读取归一化参数
    norm_param = pd.read_csv(f"normalization/{mat_prop}_minmax.csv")
    y_min = norm_param["y_min"].iloc[0]
    y_max = norm_param["y_max"].iloc[0]

    # 反归一化
    y_true = minmax_inverse(y_true_norm, y_min, y_max)
    y_pred = minmax_inverse(y_pred_norm, y_min, y_max)

    if model.classification:
        auc = roc_auc_score(y_true, y_pred)
        print(f'{mat_prop} ROC AUC ({split_name}): {auc:.4f}')
        return auc

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f'{split_name} metrics:')
    print(f'  MAE  : {mae:.4f}')
    print(f'  MSE  : {mse:.4f}')
    print(f'  RMSE : {rmse:.4f}')
    print(f'  R²   : {r2:.4f}')

    save_prediction_excel(output, mat_prop, split_name)

    os.makedirs("model_metrics", exist_ok=True)
    metrics_path = f"model_metrics/{mat_prop}_metrics.csv"
    pd.DataFrame([{
        "split": split_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }]).to_csv(
        metrics_path,
        mode='a',
        header=not os.path.exists(metrics_path),
        index=False
    )

    return mae, mse, rmse, r2


# ===============================
# 化学式分数转小数
# ===============================
def frac_to_decimal_in_formula(formula, ndigits=2):
    if not isinstance(formula, str):
        return formula

    pattern = r'([A-Z][a-z]?)(\d+)\s*/\s*(\d+)'

    def repl(m):
        return f"{m.group(1)}{round(float(m.group(2)) / float(m.group(3)), ndigits)}"

    return re.sub(pattern, repl, formula)


# ===============================
# 主程序
# ===============================
if __name__ == '__main__':

    # 1️⃣ 读取并清洗数据
    df = pd.read_excel("Unfiltered.xlsx")
    df = df[["formula", "target"]]

    df["formula"] = (
        df["formula"]
        .astype(str)
        .str.replace(r'\s+|\u200b', '', regex=True)
        .apply(frac_to_decimal_in_formula)
    )

    # 2️⃣ 固定随机划分
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=RNG_SEED
    )

    # ⭐ 只用训练集计算 Min / Max
    y_min = train_df["target"].min()
    y_max = train_df["target"].max()

    print(f"Using train-set Min-Max: min={y_min:.6f}, max={y_max:.6f}")

    # ⭐ 归一化
    train_df["target"] = minmax_scale(train_df["target"], y_min, y_max)
    test_df["target"]  = minmax_scale(test_df["target"],  y_min, y_max)

    # 保存归一化参数
    norm_dir = Path("normalization")
    norm_dir.mkdir(exist_ok=True)

    pd.DataFrame([{
        "y_min": y_min,
        "y_max": y_max
    }]).to_csv(norm_dir / "property_minmax.csv", index=False)

    # 3️⃣ 保存数据
    base_dir = Path("data/m_data/property")
    base_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(base_dir / "train.csv", index=False)
    test_df.to_csv(base_dir / "test.csv", index=False)

    # 4️⃣ 训练与评估
    data_dir = 'data/m_data'
    mat_prop = 'property'
    classification = False

    print(f'Property "{mat_prop}" selected for training')
    get_model(data_dir, mat_prop, classification)

    print('=' * 60)
    save_results(data_dir, mat_prop, classification, 'train.csv', verbose=False)
    save_results(data_dir, mat_prop, classification, 'test.csv', verbose=False)
    print('=' * 60)
