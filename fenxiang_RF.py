import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ===============================
# 1. 读取数据（包含 formula）
# ===============================
data = pd.read_excel('data/Unfiltered.xlsx')

formulas = data['formula'].values

df = data.loc[:, [
    'Type', 'MPA', 'MPB',
    'rA', 'rB',
    'IA', 'IB',
    'XA', 'XB',
    'MA', 'MB',
    'IA2', 'IB2',
    'XA2', 'XB2',
    'Co', 'Fe',
    'rAB',
    'Rp'
]].copy()

# ===============================
# 2. 按目标 Rp 分箱 + 分层拆分数据
# ===============================
target_col = 'Rp'

# —— 2.1 生成 Rp 分箱（只用于分层）——
bins = [0, 1.0, 2.0, np.inf]
df['Rp_bin'] = pd.cut(
    df[target_col],
    bins=bins,
    labels=[0, 1, 2],
    include_lowest=True
)

test_size = 0.2

# —— 2.2 第一次切分：test / remaining（分层）——
remaining_data, test_data, remaining_formula, test_formula = train_test_split(
    df,
    formulas,
    test_size=test_size,
    random_state=42,
    stratify=df['Rp_bin']
)

# —— 2.3 第二次切分：train / val（分层）——
val_size_total = 0.1
val_relative_size = val_size_total / (1 - test_size)

train_data, val_data, train_formula, val_formula = train_test_split(
    remaining_data,
    remaining_formula,
    test_size=val_relative_size,
    random_state=42,
    stratify=remaining_data['Rp_bin']
)

# —— 2.4 删除临时分箱列 ——
for split_df in [train_data, val_data, test_data]:
    split_df.drop(columns=['Rp_bin'], inplace=True)

# 最终比例检查
print(f"数据划分结果：")
print(f"训练集样本数：{len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
print(f"验证集样本数：{len(val_data)} ({len(val_data)/len(df)*100:.1f}%)")
print(f"测试集样本数：{len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")

# ===============================
# 3. 只在训练集上 fit Z-score
# ===============================
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_data)
val_scaled   = scaler.transform(val_data)
test_scaled  = scaler.transform(test_data)

train_scaled = pd.DataFrame(train_scaled, columns=df.columns[:-1])
val_scaled   = pd.DataFrame(val_scaled, columns=df.columns[:-1])
test_scaled  = pd.DataFrame(test_scaled, columns=df.columns[:-1])

# ===============================
# 4. 划分 X / y（归一化空间）
# ===============================
train_X = train_scaled.iloc[:, 0:18]
train_y = train_scaled.iloc[:, 18]

val_X   = val_scaled.iloc[:, 0:18]
val_y   = val_scaled.iloc[:, 18]

test_X  = test_scaled.iloc[:, 0:18]
test_y  = test_scaled.iloc[:, 18]

# ===============================
# 5. Random Forest 模型训练
# ===============================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
)
model.fit(train_X, train_y)

# ===============================
# 6. 预测（归一化空间）
# ===============================
train_pred_scaled = model.predict(train_X)
val_pred_scaled   = model.predict(val_X)
test_pred_scaled  = model.predict(test_X)

# ===============================
# 7. Rp 反归一化
# ===============================
def inverse_transform_rp(y_scaled, scaler, rp_index=18):
    temp = np.zeros((len(y_scaled), scaler.n_features_in_))
    temp[:, rp_index] = y_scaled
    return scaler.inverse_transform(temp)[:, rp_index]

train_y_real    = inverse_transform_rp(train_y.values, scaler)
train_pred_real = inverse_transform_rp(train_pred_scaled, scaler)

val_y_real      = inverse_transform_rp(val_y.values, scaler)
val_pred_real   = inverse_transform_rp(val_pred_scaled, scaler)

test_y_real     = inverse_transform_rp(test_y.values, scaler)
test_pred_real  = inverse_transform_rp(test_pred_scaled, scaler)

# ===============================
# 8. 评估输出
# ===============================
print("\n训练集评估结果（真实 Rp 尺度）：")
print(f"MSE  : {mean_squared_error(train_y_real, train_pred_real):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(train_y_real, train_pred_real)):.4f}")
print(f"MAE  : {mean_absolute_error(train_y_real, train_pred_real):.4f}")
print(f"R²   : {r2_score(train_y_real, train_pred_real):.4f}")

print("\n验证集评估结果（真实 Rp 尺度）：")
print(f"MSE  : {mean_squared_error(val_y_real, val_pred_real):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(val_y_real, val_pred_real)):.4f}")
print(f"MAE  : {mean_absolute_error(val_y_real, val_pred_real):.4f}")
print(f"R²   : {r2_score(val_y_real, val_pred_real):.4f}")

print("\n测试集评估结果（真实 Rp 尺度）：")
print(f"MSE  : {mean_squared_error(test_y_real, test_pred_real):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(test_y_real, test_pred_real)):.4f}")
print(f"MAE  : {mean_absolute_error(test_y_real, test_pred_real):.4f}")
print(f"R²   : {r2_score(test_y_real, test_pred_real):.4f}")

# ===============================
# 9. 逐样本误差表
# ===============================
def build_error_df(formula, y_true, y_pred, dataset_name):
    df_error = pd.DataFrame({
        'formula': formula,
        'y_true': y_true,
        'y_pred': y_pred
    })
    df_error['error'] = df_error['y_pred'] - df_error['y_true']
    df_error['abs_error'] = np.abs(df_error['error'])
    df_error['dataset'] = dataset_name
    return df_error

train_error_df = build_error_df(train_formula, train_y_real, train_pred_real, 'Train')
val_error_df   = build_error_df(val_formula, val_y_real, val_pred_real, 'Validation')
test_error_df  = build_error_df(test_formula, test_y_real, test_pred_real, 'Test')

# ===============================
# 10. 导出 Excel
# ===============================
output_file = 'RF_error_analysis（3）.xlsx'

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    train_error_df.to_excel(writer, sheet_name='Train', index=False)
    val_error_df.to_excel(writer, sheet_name='Validation', index=False)
    test_error_df.to_excel(writer, sheet_name='Test', index=False)

print(f"\n逐样本误差分析（真实 Rp）已导出至：{output_file}")
