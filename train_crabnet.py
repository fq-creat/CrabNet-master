import os
import re
import numpy as np
import pandas as pd
import torch
import sys

#from MAT2VEC import embedding_dict

sys.path.append("./CrabNet")  # 或者 "./CrabNet/crabnet" 取决于你的结构

from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

# 全局设置
compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ===============================
# 替换后的化学式分数转小数核心函数（更健壮）
# ===============================
def frac_to_decimal_in_formula(formula, ndigits=2):
    if not isinstance(formula, str):
        return formula

    # 正则匹配规则：匹配 "元素符号 + 分子 / 分母" 的模式（允许分子分母间有空格）
    # 例如：O1/2、Fe3/4、O 1 / 2 都会被正确匹配
    pattern = r'([A-Z][a-z]?)(\d+)\s*/\s*(\d+)'

    def repl(m):
        """正则替换函数：计算分数值并保留指定位数"""
        element = m.group(1)
        numerator = float(m.group(2))  # 分子
        denominator = float(m.group(3))  # 分母
        decimal_val = round(numerator / denominator, ndigits)
        # 避免出现如 0.50 这样的末尾零，简化为 0.5
        decimal_str = f"{decimal_val:.{ndigits}f}".rstrip('0').rstrip('.')
        return f"{element}{decimal_str}"

    # 执行正则替换
    processed_formula = re.sub(pattern, repl, formula)
    # 额外清理可能的多余空格（兜底处理）
    processed_formula = re.sub(r'\s+', '', processed_formula)
    return processed_formula


# ===============================
# 原有核心功能函数（仅调整化学式处理调用）
# ===============================
def get_model(data_dir, mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # 加载CrabNet模型架构
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)

    # 加载预训练权重（如有）
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    # 分类任务设置
    if classification:
        model.classification = True

    # 加载训练和验证数据路径
    train_data = f'{data_dir}/{mat_prop}/train.csv'
    val_data = f'{data_dir}/{mat_prop}/val.csv'

    # 检查验证集是否存在（修正原代码异常捕获无效的问题）
    if not os.path.exists(val_data):
        raise FileNotFoundError(
            f'验证集文件不存在！请确保 {data_dir}/{mat_prop} 目录下有 val.csv 文件'
        )

    # 计算合理的batch size
    # data_size = pd.read_csv(train_data).shape[0]
    # batch_size = 2 ** round(np.log2(data_size) - 4)
    # batch_size = max(2 ** 7, min(batch_size, 2 ** 12))  # 简化边界判断
    FIXED_BATCH_SIZE =128
    batch_size = FIXED_BATCH_SIZE
    # 加载训练数据
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    # 加载验证数据
    model.load_data(val_data, batch_size=batch_size)

    # 训练模型
    model.fit(epochs=500, losscurve=False)
    # 保存模型
    model.save_network()
    return model


def save_prediction_excel(output, mat_prop, split_name):
    import os
    from pathlib import Path

    # 解析模型输出
    y_true, y_pred, formulae, _ = output

    # 构建结果DataFrame
    df = pd.DataFrame({
        "formula": formulae,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    # 计算误差列
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    # 标记数据集类型
    df["dataset"] = split_name

    # 创建保存目录
    save_dir = Path("model_predictions_excel")
    save_dir.mkdir(exist_ok=True)
    # 保存Excel文件（每个模型一个文件，包含所有数据集结果）
    save_path = save_dir / f"{mat_prop}_predictions.xlsx"

    # 若文件已存在（比如先存了train，再存val/test），追加到同一个文件的不同sheet
    if os.path.exists(save_path):
        with pd.ExcelWriter(save_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=split_name, index=False)
    else:
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=split_name, index=False)

    print(f"✅ {split_name}集预测结果已保存到Excel: {save_path} (sheet: {split_name})")
def to_csv(output, save_name):
    # 解析预测结果并保存为CSV
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    save_path = 'model_predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(data_dir, mat_prop, classification, file_name, verbose=True):
    # 加载已保存的模型
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)
    model.load_network(f'{mat_prop}.pth')

    # 分类任务设置
    if classification:
        model.classification = True

    # 加载待预测数据
    data = f'{data_dir}/{mat_prop}/{file_name}'
    model.load_data(data, batch_size=128, train=False)
    return model


def get_results(model):
    # 执行预测并返回结果
    output = model.predict(model.data_loader)
    return model, output


def save_results(data_dir, mat_prop, classification, file_name, verbose=True):
    # 加载模型并计算评估指标
    model = load_model(data_dir, mat_prop, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    # 获取真实值和预测值
    y_true = output[0]
    y_pred = output[1]

    # 根据任务类型计算指标
    if model.classification:
        auc = roc_auc_score(y_true, y_pred)
        print(f'{mat_prop} ROC AUC: {auc:0.3f}')
        metrics = {'auc': auc}
    else:
        # 回归任务指标计算
        mae = np.abs(y_true - y_pred).mean()
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # 打印指标
        print(f'{mat_prop} 数据集: {file_name}')
        print(f'  MAE:  {mae:0.4f}')
        print(f'  MSE:  {mse:0.4f}')
        print(f'  RMSE: {rmse:0.4f}')
        print(f'  R²:   {r2:0.4f}')
        print('-' * 30)

        metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

    # 保存预测结果
    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    to_csv(output, fname)
    split_name = file_name.replace(".csv", "")
    save_prediction_excel(output, mat_prop, split_name)
    return model, metrics


def preprocess_excel_to_csv(excel_path, mat_prop, test_size=0.2, val_size=0.1):
    # 预处理Excel数据：清洗化学式 + 按目标分箱分层划分数据集 + 保存为CSV
    df = pd.read_excel(excel_path, sheet_name=0)

    # ========== 化学式清洗 ==========
    df['formula'] = (
        df['formula']
        .astype(str)
        .str.replace(r'\s+|\u200b', '', regex=True)
        .apply(frac_to_decimal_in_formula)
    )

    # ========== 目标列 ==========
    target_col = df.columns[1]

    # ========== 按目标分箱（只用于分层） ==========
    bins = [0, 1.0, 2.0, np.inf]
    df["y_bin"] = pd.cut(
        df[target_col],
        bins=bins,
        labels=[0, 1, 2],
        include_lowest=True
    )

    # ========== 第一次切分：train_val / test（分层） ==========
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RNG_SEED,
        stratify=df["y_bin"]
    )

    # ========== 第二次切分：train / val（分层） ==========
    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=RNG_SEED,
        stratify=train_val_df["y_bin"]
    )

    # ========== 删除临时分箱列 ==========
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["y_bin"], inplace=True)

    # ========== 创建保存目录 ==========
    base_dir = 'data'
    prop_dir = os.path.join(base_dir, mat_prop)
    os.makedirs(prop_dir, exist_ok=True)

    # ========== 保存CSV ==========
    train_df.to_csv(os.path.join(prop_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(prop_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(prop_dir, 'test.csv'), index=False)

    return base_dir, mat_prop



# ===============================
# 主程序
# ===============================
if __name__ == '__main__':
    excel_path = 'Unfiltered.xlsx'  # 输入Excel文件路径
    mat_prop = 'example_materials_property'  # 材料属性名称
    classification = False  # 是否为分类任务（False=回归，True=分类）
    train = True  # 是否训练模型

    # 预处理Excel数据
    if excel_path.endswith(('.xlsx', '.xls')):
        data_dir, mat_prop = preprocess_excel_to_csv(
            excel_path,
            mat_prop,
            test_size=0.2,  # 测试集比例
            val_size=0.1  # 验证集比例
        )
    else:
        data_dir = excel_path

    # 训练模型
    if train:
        model = get_model(data_dir, mat_prop, classification, verbose=True)

    # 打印分隔符和评估结果
    cutter = '====================================================='
    first = " " * ((len(cutter) - len(mat_prop)) // 2) + " " * int((len(mat_prop) + 1) % 2)
    last = " " * ((len(cutter) - len(mat_prop)) // 2)
    print(f'{first}{mat_prop}{last}')

    # 评估训练集
    print('\n训练集性能:')
    model_train, metrics_train = save_results(data_dir, mat_prop, classification,
                                              'train.csv', verbose=False)
    # 评估验证集
    print('\n验证集性能:')
    model_val, metrics_val = save_results(data_dir, mat_prop, classification,
                                          'val.csv', verbose=False)
    # 评估测试集
    print('\n测试集性能:')
    model_test, metrics_test = save_results(data_dir, mat_prop, classification,
                                            'test.csv', verbose=False)
