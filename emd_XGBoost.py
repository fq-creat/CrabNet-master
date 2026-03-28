# =========================
# Z-score 归一化 + 按目标分箱分层的 Train / Val / Test
# =========================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# 1. 读取 encoder 数据
# =========================
data = pd.read_excel('model_embeddings/example_materials_property_all_encoder_frac_weighted.xlsx')

formulas = data['formula'].values

# 自动识别 embedding 特征
feature_cols = [c for c in data.columns if c.startswith("emb_")]

df = data[feature_cols + ['target']].copy()

# 为兼容旧代码，把 target 改名为 Rp
df.rename(columns={'target': 'Rp'}, inplace=True)


def _can_stratify(y_bin):
    vc = y_bin.value_counts(dropna=False)
    return y_bin.nunique(dropna=True) >= 2 and len(vc) > 0 and (vc.min() >= 2)


def _group_split_indices(formula_series, target_series, test_size, random_state):
    grouped = pd.DataFrame({
        'formula': formula_series,
        'target': target_series
    }).groupby('formula', as_index=False)['target'].mean()
    grouped['y_bin'] = pd.qcut(grouped['target'], q=3, labels=False, duplicates='drop')
    stratify = grouped['y_bin'] if _can_stratify(grouped['y_bin']) else None
    train_groups, test_groups = train_test_split(
        grouped['formula'],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    train_groups = set(train_groups.tolist())
    test_groups = set(test_groups.tolist())
    train_mask = pd.Series(formula_series).isin(train_groups).values
    test_mask = pd.Series(formula_series).isin(test_groups).values
    return train_mask, test_mask

dataset_col = data['dataset'].astype(str).str.lower() if 'dataset' in data.columns else None
has_explicit_split = dataset_col is not None and {'train', 'val', 'test'}.issubset(set(dataset_col))

if has_explicit_split:
    train_mask = dataset_col == 'train'
    val_mask = dataset_col == 'val'
    test_mask = dataset_col == 'test'

    train_data = df.loc[train_mask].reset_index(drop=True)
    val_data = df.loc[val_mask].reset_index(drop=True)
    test_data = df.loc[test_mask].reset_index(drop=True)

    train_formula = formulas[train_mask]
    val_formula = formulas[val_mask]
    test_formula = formulas[test_mask]
else:
    remain_mask, test_mask = _group_split_indices(formulas, df['Rp'].values, 0.2, 42)
    remain_data = df.loc[remain_mask].reset_index(drop=True)
    test_data = df.loc[test_mask].reset_index(drop=True)
    remain_formula = formulas[remain_mask]
    test_formula = formulas[test_mask]

    val_relative_size = 0.1 / 0.8  # 0.125
    train_mask_inner, val_mask = _group_split_indices(
        remain_formula,
        remain_data['Rp'].values,
        val_relative_size,
        42
    )
    train_data = remain_data.loc[train_mask_inner].reset_index(drop=True)
    val_data = remain_data.loc[val_mask].reset_index(drop=True)
    train_formula = remain_formula[train_mask_inner]
    val_formula = remain_formula[val_mask]

TrainData, ValData, TestData = train_data, val_data, test_data
Train_formula, Val_formula, Test_formula = train_formula, val_formula, test_formula

print("数据划分完成（按 Rp 分箱分层）：")
print(f"Train: {len(TrainData)} | Val: {len(ValData)} | Test: {len(TestData)}")

# =========================
# 5. Z-score（仅在训练集 fit）
# =========================
scaler = StandardScaler()

TrainData_scaled = scaler.fit_transform(TrainData)
ValData_scaled   = scaler.transform(ValData)
TestData_scaled  = scaler.transform(TestData)

columns = TrainData.columns

TrainData_scaled = pd.DataFrame(TrainData_scaled, columns=columns)
ValData_scaled   = pd.DataFrame(ValData_scaled, columns=columns)
TestData_scaled  = pd.DataFrame(TestData_scaled, columns=columns)

# =========================
# 6. X / y
# =========================
feature_dim = len(feature_cols)

Train_X = TrainData_scaled.iloc[:, :feature_dim]
Train_y = TrainData_scaled.iloc[:, feature_dim]

Val_X   = ValData_scaled.iloc[:, :feature_dim]
Val_y   = ValData_scaled.iloc[:, feature_dim]

Test_X  = TestData_scaled.iloc[:, :feature_dim]
Test_y  = TestData_scaled.iloc[:, feature_dim]

# =========================
# 7. XGBoost
# =========================
model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1
)

model.fit(Train_X, Train_y)

# =========================
# 8. 预测
# =========================
Train_pred_scaled = model.predict(Train_X)
Val_pred_scaled   = model.predict(Val_X)
Test_pred_scaled  = model.predict(Test_X)

# =========================
# 9. Rp 反归一化
# =========================
def inverse_transform_rp(y_scaled, scaler, rp_index):

    tmp = np.zeros((len(y_scaled), scaler.n_features_in_))
    tmp[:, rp_index] = y_scaled

    return scaler.inverse_transform(tmp)[:, rp_index]


rp_index = feature_dim

Train_y_real = inverse_transform_rp(Train_y.values, scaler, rp_index)
Train_pred_real = inverse_transform_rp(Train_pred_scaled, scaler, rp_index)

Val_y_real = inverse_transform_rp(Val_y.values, scaler, rp_index)
Val_pred_real = inverse_transform_rp(Val_pred_scaled, scaler, rp_index)

Test_y_real = inverse_transform_rp(Test_y.values, scaler, rp_index)
Test_pred_real = inverse_transform_rp(Test_pred_scaled, scaler, rp_index)

# =========================
# 10. 评估
# =========================
def evaluate(name, y_true, y_pred):

    print(f"\n{name} 评估：")
    print(f"MSE : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAE : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R²  : {r2_score(y_true, y_pred):.4f}")

evaluate("Train", Train_y_real, Train_pred_real)
evaluate("Val",   Val_y_real,   Val_pred_real)
evaluate("Test",  Test_y_real,  Test_pred_real)

# =========================
# 11. 逐样本误差导出
# =========================
def build_error_df(formula, y_true, y_pred, name):

    df_e = pd.DataFrame({
        'formula': formula,
        'y_true': y_true,
        'y_pred': y_pred
    })

    df_e['error'] = df_e['y_pred'] - df_e['y_true']
    df_e['abs_error'] = np.abs(df_e['error'])
    df_e['dataset'] = name

    return df_e


train_df = build_error_df(Train_formula, Train_y_real, Train_pred_real, 'Train')
val_df   = build_error_df(Val_formula,   Val_y_real,   Val_pred_real,   'Val')
test_df  = build_error_df(Test_formula,  Test_y_real,  Test_pred_real,  'Test')

# 保持小数位与原脚本一致
train_df = train_df.round(4)
val_df   = val_df.round(4)
test_df  = test_df.round(4)

with pd.ExcelWriter('XGBoost_error_analysis_stratified_encoder.xlsx') as writer:
    train_df.to_excel(writer, sheet_name='Train', index=False)
    val_df.to_excel(writer,   sheet_name='Val',   index=False)
    test_df.to_excel(writer,  sheet_name='Test',  index=False)

print("\n误差分析文件已导出（按 Rp 分箱分层）")
