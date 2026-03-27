import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append("./CrabNet")  # 保持与你当前工程一致

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
# 工具函数
# ===============================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("布尔参数请使用 true/false")


def frac_to_decimal_in_formula(formula, ndigits=2):
    """将化学式中的分数字符串替换为小数（例如 O1/2 -> O0.5）。"""
    if not isinstance(formula, str):
        return formula

    pattern = r'([A-Z][a-z]?)(\d+)\s*/\s*(\d+)'

    def repl(m):
        element = m.group(1)
        numerator = float(m.group(2))
        denominator = float(m.group(3))
        decimal_val = round(numerator / denominator, ndigits)
        decimal_str = f"{decimal_val:.{ndigits}f}".rstrip('0').rstrip('.')
        return f"{element}{decimal_str}"

    processed_formula = re.sub(pattern, repl, formula)
    processed_formula = re.sub(r'\s+', '', processed_formula)
    return processed_formula


# ===============================
# 模型相关函数
# ===============================
def get_model(data_dir, mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True, epochs=500):
    """加载并训练 CrabNet。"""
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
    val_data = f'{data_dir}/{mat_prop}/val.csv'

    if not os.path.exists(train_data):
        raise FileNotFoundError(f'训练集文件不存在: {train_data}')
    if not os.path.exists(val_data):
        raise FileNotFoundError(f'验证集文件不存在: {val_data}')

    if batch_size is None:
        # 与你原脚本保持一致：固定 batch_size=1
        # 如果你想加速，可改成动态策略
        batch_size = 16

    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    model.fit(epochs=epochs, losscurve=False)
    model.save_network()
    return model


def load_model(data_dir, mat_prop, classification, file_name, verbose=True):
    """加载已保存模型并读取预测数据。"""
    model = Model(
        CrabNet(compute_device=compute_device).to(compute_device),
        model_name=f'{mat_prop}',
        verbose=verbose
    )
    model.load_network(f'{mat_prop}.pth')

    if classification:
        model.classification = True

    data = f'{data_dir}/{mat_prop}/{file_name}'
    if not os.path.exists(data):
        raise FileNotFoundError(f'数据文件不存在: {data}')

    model.load_data(data, batch_size=16, train=False)
    return model


def get_results(model):
    output = model.predict(model.data_loader)
    return model, output


def to_csv(output, save_name):
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    save_path = 'model_predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def save_prediction_excel(output, mat_prop, split_name):
    y_true, y_pred, formulae, _ = output
    df = pd.DataFrame({
        "formula": formulae,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    df["dataset"] = split_name

    save_dir = "model_predictions_excel"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{mat_prop}_predictions.xlsx")

    if os.path.exists(save_path):
        with pd.ExcelWriter(save_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=split_name, index=False)
    else:
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=split_name, index=False)

    print(f"✅ {split_name}集预测结果已保存到Excel: {save_path} (sheet: {split_name})")


def save_results(data_dir, mat_prop, classification, file_name, verbose=True):
    model = load_model(data_dir, mat_prop, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    y_true = output[0]
    y_pred = output[1]

    if model.classification:
        auc = roc_auc_score(y_true, y_pred)
        print(f'{mat_prop} ROC AUC: {auc:0.3f}')
        metrics = {'auc': auc}
    else:
        mae = np.abs(y_true - y_pred).mean()
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f'{mat_prop} 数据集: {file_name}')
        print(f'  MAE:  {mae:0.4f}')
        print(f'  MSE:  {mse:0.4f}')
        print(f'  RMSE: {rmse:0.4f}')
        print(f'  R²:   {r2:0.4f}')
        print('-' * 30)

        metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    to_csv(output, fname)
    split_name = file_name.replace(".csv", "")
    save_prediction_excel(output, mat_prop, split_name)
    return model, metrics


def save_encoder_embeddings_csv(data_dir, mat_prop, classification, file_name,
                                pooling='frac_weighted', verbose=True):
    """
    导出 encoder composition 向量（不是 output_nn 后的向量）。
    pooling:
      - 'frac_weighted': 按元素分数加权平均
      - 'mean': 有效元素位点均值
    """
    model = load_model(data_dir, mat_prop, classification, file_name, verbose=verbose)
    model.model.eval()

    all_formulae = []
    all_targets = []
    all_embeddings = []

    with torch.no_grad():
        for data in model.data_loader:
            X, y, formula = data
            src, frac = X.squeeze(-1).chunk(2, dim=1)

            src = src.to(model.compute_device, dtype=torch.long, non_blocking=True)
            frac = frac.to(model.compute_device, dtype=torch.float32, non_blocking=True)

            # element-level encoder 输出: [batch, n_elements, d_model]
            elem_emb = model.model.encoder(src, frac)
            valid_mask = (src != 0).float()

            if pooling == 'frac_weighted':
                weights = frac * valid_mask
                denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
                weights = weights / denom
                comp_emb = (elem_emb * weights.unsqueeze(-1)).sum(dim=1)
            elif pooling == 'mean':
                denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                comp_emb = (elem_emb * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                raise ValueError("pooling 必须是 'frac_weighted' 或 'mean'")

            all_embeddings.append(comp_emb.cpu().numpy())
            all_formulae.extend(list(formula))
            all_targets.extend(y.view(-1).cpu().numpy().astype('float32').tolist())

    embeddings = np.vstack(all_embeddings)
    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, 'formula', all_formulae)
    df.insert(1, 'target', all_targets)

    save_dir = 'model_embeddings'
    os.makedirs(save_dir, exist_ok=True)
    split_name = file_name.replace('.csv', '')
    save_path = os.path.join(save_dir, f'{mat_prop}_{split_name}_encoder_{pooling}.csv')
    df.to_csv(save_path, index=False)

    print(f'✅ 已保存 {split_name} 的 encoder 向量到: {save_path}')
    return save_path


def merge_embedding_csvs_to_excel(csv_paths, mat_prop, pooling='frac_weighted'):
    merged_parts = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        merged_parts.append(df)

    merged_df = pd.concat(merged_parts, ignore_index=True)

    save_dir = 'model_embeddings'
    os.makedirs(save_dir, exist_ok=True)
    merged_csv_path = os.path.join(save_dir, f'{mat_prop}_all_encoder_{pooling}.csv')
    merged_xlsx_path = os.path.join(save_dir, f'{mat_prop}_all_encoder_{pooling}.xlsx')

    merged_df.to_csv(merged_csv_path, index=False)
    merged_df.to_excel(merged_xlsx_path, index=False)

    print(f'✅ 已合并并保存 CSV: {merged_csv_path}')
    print(f'✅ 已转换并保存 Excel: {merged_xlsx_path}')
    return merged_csv_path, merged_xlsx_path


# ===============================
# 数据准备函数（Excel模式）
# ===============================
def preprocess_excel_to_csv(excel_path, mat_prop, test_size=0.2, val_size=0.1):
    """
    预处理 Excel：
    - 清洗化学式
    - 分层划分 train/val/test
    - 保存到 data/<mat_prop>/{train,val,test}.csv
    """
    df = pd.read_excel(excel_path, sheet_name=0)

    df['formula'] = (
        df['formula']
        .astype(str)
        .str.replace(r'\s+|\u200b', '', regex=True)
        .apply(frac_to_decimal_in_formula)
    )

    target_col = df.columns[1]

    # 分箱分层（回归任务常见技巧）
    bins = [0, 1.0, 2.0, np.inf]
    df["y_bin"] = pd.cut(df[target_col], bins=bins, labels=[0, 1, 2], include_lowest=True)

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=RNG_SEED, stratify=df["y_bin"]
    )

    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=RNG_SEED,
        stratify=train_val_df["y_bin"]
    )

    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["y_bin"], inplace=True)

    # 统一列名为 formula/target（与 CrabNet 习惯一致）
    train_df = train_df.rename(columns={target_col: "target"})
    val_df = val_df.rename(columns={target_col: "target"})
    test_df = test_df.rename(columns={target_col: "target"})

    base_dir = 'data'
    prop_dir = os.path.join(base_dir, mat_prop)
    os.makedirs(prop_dir, exist_ok=True)

    train_df[['formula', 'target']].to_csv(os.path.join(prop_dir, 'train.csv'), index=False)
    val_df[['formula', 'target']].to_csv(os.path.join(prop_dir, 'val.csv'), index=False)
    test_df[['formula', 'target']].to_csv(os.path.join(prop_dir, 'test.csv'), index=False)

    return base_dir, mat_prop


# ===============================
# 数据准备函数（Matbench模式）
# ===============================
def prepare_matbench_task_to_csv(matbench_task='matbench_perovskites',
                                 mat_prop='castelli',
                                 fold=0,
                                 val_size=0.1,
                                 seed=42):
    """
    从 Matbench 指定任务读取一个 fold，保存为:
      data/<mat_prop>/train.csv
      data/<mat_prop>/val.csv
      data/<mat_prop>/test.csv

    说明：
    - matbench 提供的是 train+val 与 test；
    - 这里把 train+val 再切分一部分做 val（便于与你现有训练流程兼容）。
    """
    try:
        from matbench.bench import MatbenchBenchmark
    except Exception as e:
        raise ImportError(
            "未安装 matbench，请先安装：pip install matbench"
        ) from e

    mb = MatbenchBenchmark(subset=[matbench_task], autoload=False)
    if len(mb.tasks) == 0:
        raise ValueError(f'未找到 Matbench 任务: {matbench_task}')

    task = mb.tasks[0]
    task.load()

    X_trainval, y_trainval = task.get_train_and_val_data(fold)
    X_test, y_test = task.get_test_data(fold, include_target=True)

    # 某些任务返回的是 Composition 对象，统一转字符串
    df_trainval = pd.DataFrame({
        'formula': pd.Series(X_trainval).astype(str).values,
        'target': pd.Series(y_trainval).values
    })
    df_test = pd.DataFrame({
        'formula': pd.Series(X_test).astype(str).values,
        'target': pd.Series(y_test).values
    })

    # 清洗化学式（与 Excel 流程保持一致）
    df_trainval['formula'] = (
        df_trainval['formula'].astype(str).str.replace(r'\s+|\u200b', '', regex=True).apply(frac_to_decimal_in_formula)
    )
    df_test['formula'] = (
        df_test['formula'].astype(str).str.replace(r'\s+|\u200b', '', regex=True).apply(frac_to_decimal_in_formula)
    )

    # 回归任务默认非分层；如需分层可加分箱逻辑
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_size, random_state=seed
    )

    base_dir = 'data'
    prop_dir = os.path.join(base_dir, mat_prop)
    os.makedirs(prop_dir, exist_ok=True)

    df_train[['formula', 'target']].to_csv(os.path.join(prop_dir, 'train.csv'), index=False)
    df_val[['formula', 'target']].to_csv(os.path.join(prop_dir, 'val.csv'), index=False)
    df_test[['formula', 'target']].to_csv(os.path.join(prop_dir, 'test.csv'), index=False)

    print(f'✅ Matbench任务 {matbench_task} (fold={fold}) 已保存到: {prop_dir}')
    return base_dir, mat_prop


# ===============================
# 主程序
# ===============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrabNet 训练 + encoder embedding 导出')

    # 数据来源
    parser.add_argument('--source', type=str, default='matbench', choices=['excel', 'matbench'],
                        help='数据来源：excel 或 matbench')

    # Excel 模式参数
    parser.add_argument('--excel_path', type=str, default='filtered.xlsx',
                        help='Excel 输入路径（source=excel 时使用）')

    # Matbench 模式参数
    parser.add_argument('--matbench_task', type=str, default='matbench_perovskites',
                        help='Matbench 任务名，例如 matbench_perovskites')
    parser.add_argument('--fold', type=int, default=0, help='Matbench fold 编号')
    parser.add_argument('--val_size', type=float, default=0.1, help='从 trainval 切给 val 的比例')

    # 通用参数
    parser.add_argument('--mat_prop', type=str, default='castelli',
                        help='本地保存用的数据集名（目录名）')
    parser.add_argument('--classification', type=str2bool, default=False,
                        help='是否分类任务')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='是否训练模型')
    parser.add_argument('--pooling', type=str, default='frac_weighted',
                        choices=['frac_weighted', 'mean'],
                        help='encoder pooling 方式')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    args = parser.parse_args()

    # 1) 准备数据
    if args.source == 'excel':
        if not args.excel_path.endswith(('.xlsx', '.xls')):
            raise ValueError("source=excel 时，--excel_path 必须是 xlsx/xls 文件")
        data_dir, mat_prop = preprocess_excel_to_csv(
            excel_path=args.excel_path,
            mat_prop=args.mat_prop,
            test_size=0.2,
            val_size=args.val_size
        )
    else:
        data_dir, mat_prop = prepare_matbench_task_to_csv(
            matbench_task=args.matbench_task,
            mat_prop=args.mat_prop,   # 推荐 castelli，方便与你现有脚本命名一致
            fold=args.fold,
            val_size=args.val_size,
            seed=RNG_SEED
        )

    # 2) 训练
    if args.train:
        _ = get_model(
            data_dir=data_dir,
            mat_prop=mat_prop,
            classification=args.classification,
            batch_size=args.batch_size,
            verbose=True,
            epochs=args.epochs
        )

    # 3) 评估 + 预测保存
    cutter = '=' * 53
    first = " " * ((len(cutter) - len(mat_prop)) // 2) + " " * int((len(mat_prop) + 1) % 2)
    last = " " * ((len(cutter) - len(mat_prop)) // 2)
    print(f'{first}{mat_prop}{last}')

    print('\n训练集性能:')
    _, _ = save_results(data_dir, mat_prop, args.classification, 'train.csv', verbose=False)

    print('\n验证集性能:')
    _, _ = save_results(data_dir, mat_prop, args.classification, 'val.csv', verbose=False)

    print('\n测试集性能:')
    _, _ = save_results(data_dir, mat_prop, args.classification, 'test.csv', verbose=False)

    # 4) 导出 encoder 向量
    print('\n导出 encoder composition 向量:')
    train_emb_csv = save_encoder_embeddings_csv(
        data_dir, mat_prop, args.classification, 'train.csv',
        pooling=args.pooling, verbose=False
    )
    val_emb_csv = save_encoder_embeddings_csv(
        data_dir, mat_prop, args.classification, 'val.csv',
        pooling=args.pooling, verbose=False
    )
    test_emb_csv = save_encoder_embeddings_csv(
        data_dir, mat_prop, args.classification, 'test.csv',
        pooling=args.pooling, verbose=False
    )

    # 5) 合并 embedding
    print('\n合并 embedding CSV 并导出 Excel:')
    merge_embedding_csvs_to_excel(
        [train_emb_csv, val_emb_csv, test_emb_csv],
        mat_prop,
        pooling=args.pooling
    )
