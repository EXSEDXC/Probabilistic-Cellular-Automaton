"""
完整版本：可解释的非线性拟合（PyTorch）
功能：
 - 自动加载 ./results/group*/ 下的各类 summary 与 params.json
 - 构建可解释的非线性模型（x, x^2, log(x), exp(-x), tanh(x)）
 - 训练双输出（survival, density），评估并绘图
 - 自动提取及反标准化系数，支持按 top_k 或 threshold 自动简化公式
 - 输出预测 CSV、诊断图
保存：nonlinear_sigmoid_fit_results.csv
"""

import os
import json
from pathlib import Path
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix,r2_score

# ----------------------------
# Config - change as needed
# ----------------------------
RESULTS_ROOT = "./results"   
FEATURE_NAMES = ["p0", "P_birth", "P_death_base", "alpha", "N_birth_max", "N_death_min"]

# Which nonlinear basis to include; order matters for formula printing
NONLINEAR_FUNCS = ["x", "x^2", "log", "exp_neg", "tanh"]

DEFAULT_EPOCHS = 400
DEFAULT_LR = 1e-2
DEFAULT_BATCH = 32

# ----------------------------
# Utilities: robust file loading
# ----------------------------
def find_group_paths(root=RESULTS_ROOT):
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"{root} not found")
    groups = []
    for p in sorted(rootp.iterdir()):
        if p.is_dir():
            if (p / "params.json").exists() or p.name.lower().startswith("group"):
                groups.append(p)
    return groups

def try_read_csv(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                continue
    return None

def load_params(p: Path):
    j = p / "params.json"
    if not j.exists():
        return {}
    try:
        return json.loads(j.read_text(encoding="utf-8"))
    except Exception:
        try:
            with open(j, "r", encoding="utf-8") as f: return json.load(f)
        except:
            return {}

def load_one_group(path: Path):
    surv = try_read_csv([path / "survival_summary.csv", path / "survival.csv", path / "survival_summary"])
    dens = try_read_csv([path / "density_summary.csv", path / "density.csv", path / "stable_density.csv"])
    cluster = try_read_csv([path / "cluster_summary.csv", path / "cluster.csv"])
    mc = try_read_csv([path / "montecarlo_results.csv", path / "montecarlo.csv"])

    params = load_params(path)
    if surv is None:
        if mc is not None and "p0" in mc.columns and "final_density" in mc.columns:
            grp = mc.groupby("p0")["final_density"].apply(lambda x: (x > 0).mean()).reset_index()
            grp.columns = ["p0", "survival_prob"]
            surv = grp
    if surv is None:
        return None

    if "p0" not in surv.columns:
        return None

    df = surv.copy()
    if "survival_prob" not in df.columns:
        for c in df.columns:
            if "surviv" in c.lower():
                df = df.rename(columns={c: "survival_prob"})
                break
    if dens is not None and "p0" in dens.columns:
        dens_sub = dens.copy()
        if "stable_density_mean" in dens_sub.columns:
            dens_sub = dens_sub[["p0", "stable_density_mean"]]
        else:
            numeric_cols = [c for c in dens_sub.columns if c != "p0" and np.issubdtype(dens_sub[c].dtype, np.number)]
            if numeric_cols:
                dens_sub = dens_sub[["p0", numeric_cols[0]]]
                dens_sub.columns = ["p0", "stable_density_mean"]
            else:
                dens_sub = None
        if dens_sub is not None:
            df = pd.merge(df, dens_sub, on="p0", how="left")
    else:
        if cluster is not None:
            numeric_cols = [c for c in cluster.columns if np.issubdtype(cluster[c].dtype, np.number)]
            if numeric_cols:
                df["stable_density_mean"] = cluster[numeric_cols[0]].iloc[0]
            else:
                df["stable_density_mean"] = 0.0
        else:
            df["stable_density_mean"] = 0.0
    for k, v in params.items():
        df[k] = v
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = np.nan
    for f in FEATURE_NAMES:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)

    df["p0"] = pd.to_numeric(df["p0"], errors="coerce")
    if "survival_prob" not in df.columns:
        numeric_cols = [c for c in df.columns if c != "p0" and np.issubdtype(df[c].dtype, np.number)]
        if numeric_cols:
            df = df.rename(columns={numeric_cols[0]: "survival_prob"})
        else:
            df["survival_prob"] = 0.0
    df["survival_prob"] = pd.to_numeric(df["survival_prob"], errors="coerce").fillna(0.0)
    df["stable_density_mean"] = pd.to_numeric(df["stable_density_mean"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["p0"]).reset_index(drop=True)
    return df

def load_all_results(root=RESULTS_ROOT):
    groups = find_group_paths(root)
    rows = []
    for g in groups:
        df = load_one_group(g)
        if df is not None and len(df) > 0:
            rows.append(df)
    if not rows:
        raise RuntimeError(f"No usable data found under {root}")
    big = pd.concat(rows, ignore_index=True, sort=False)
    big["survival_prob"] = pd.to_numeric(big["survival_prob"], errors="coerce").fillna(0.0)
    big["stable_density_mean"] = pd.to_numeric(big["stable_density_mean"], errors="coerce").fillna(0.0)
    for f in FEATURE_NAMES:
        big[f] = pd.to_numeric(big[f], errors="coerce").fillna(0.0)
    return big
# ----------------------------
# Dataset
# ----------------------------
class SimDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler):
        self.X = scaler.transform(df[FEATURE_NAMES].values.astype(float)).astype(np.float32)
        self.y_surv = df["survival_prob"].astype(float).values.astype(np.float32)
        self.y_dens = df["stable_density_mean"].astype(float).values.astype(np.float32)
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y_surv[idx], dtype=torch.float32), torch.tensor(self.y_dens[idx], dtype=torch.float32)
# ----------------------------
# Model: explicit nonlinear basis (all bases concatenated)
# ----------------------------
class ExplainableNonlinearModel(nn.Module):
    def __init__(self, n_features: int, funcs):
        """
        n_features: number of original features
        funcs: list of strings among {"x","x^2","log","exp_neg","tanh"}
        The forward will compute each func on standardized x and concatenate then linear map to two outputs
        """
        super().__init__()
        self.nf = n_features
        self.funcs = funcs.copy()
        self.total_dim = len(funcs) * n_features
        self.linear = nn.Linear(self.total_dim, 2)  
    def forward(self, x):
        blocks = []
        eps = 1e-9
        for f in self.funcs:
            if f == "x":
                blocks.append(x)
            elif f == "x^2":
                blocks.append(x * x)
            elif f == "log":
                blocks.append(torch.log(torch.clamp(x + 1e-6, min=1e-6)))
            elif f == "exp_neg":
                blocks.append(torch.exp(-x))
            elif f == "tanh":
                blocks.append(torch.tanh(x))
            else:
                raise ValueError("Unknown func: " + str(f))
        Xext = torch.cat(blocks, dim=1)
        out = self.linear(Xext)
        logit_s = out[:, 0]
        logit_d = out[:, 1]
        p_s = torch.sigmoid(logit_s)
        p_d = torch.sigmoid(logit_d)
        return p_s, p_d, logit_s, logit_d
# ----------------------------
# Training and evaluation
# ----------------------------
def train_model(model, dataset: SimDataset, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH, device="cpu", lambda_density=1.0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, ys, yd in loader:
            xb = xb.to(device); ys = ys.to(device); yd = yd.to(device)
            ps, pd, _, _ = model(xb)
            loss = bce(ps, ys) + lambda_density * mse(pd, yd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
        if epoch % max(1, epochs//8) == 0 or epoch <= 5:
            print(f"Epoch {epoch}/{epochs}  avg_loss={total_loss / max(1,n):.6f}")
    return model

def evaluate_model(model, dataset: SimDataset, device="cpu"):
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(dataset.X).to(device)
        ps, pd, logit_s, logit_d = model(X)
        ps = ps.cpu().numpy(); pd = pd.cpu().numpy()
    ys = dataset.y_surv; yd = dataset.y_dens
    mse_surv = mean_squared_error(ys, ps)
    mse_dens = mean_squared_error(yd, pd)
    r2_surv = r2_score(ys, ps)
    r2_dens = r2_score(yd, pd)

    return {
        "mse_surv": mse_surv,
        "mse_dens": mse_dens,
        "r2_surv": r2_surv,
        "r2_dens": r2_dens,
        "ps": ps,
        "pd": pd
    }
# ----------------------------
# Formula extraction & simplification
# ----------------------------
def extract_weight_terms(model: ExplainableNonlinearModel, scaler: StandardScaler, feature_names, funcs):
    """
    返回 dict mapping term_key -> metadata:
    {
      'key': 'log(p0)',
      'base': 'p0',
      'func': 'log',           # one of funcs strings like 'x','x^2','log','exp_neg','tanh'
      'coef_surv': ...,
      'coef_dens': ...,
      'mu': ...,
      'sigma': ...,
      'flat_index': int
    }
    """
    W = model.linear.weight.detach().cpu().numpy()  
    b = model.linear.bias.detach().cpu().numpy()    
    mu = scaler.mean_
    sigma = scaler.scale_
    n = len(feature_names)
    terms = {}
    for fi, func in enumerate(funcs):
        for i, fname in enumerate(feature_names):
            idx = fi * n + i
            coef_surv = float(W[0, idx])
            coef_dens = float(W[1, idx])
            if func == "x":
                key = f"{fname}"
                func_name = "x"
                base = fname
            elif func == "x^2":
                key = f"{fname}^2"
                func_name = "x^2"
                base = fname
            elif func == "log":
                key = f"log({fname})"
                func_name = "log"
                base = fname
            elif func == "exp_neg":
                key = f"exp(-{fname})"
                func_name = "exp_neg"
                base = fname
            elif func == "tanh":
                key = f"tanh({fname})"
                func_name = "tanh"
                base = fname
            else:
                key = f"{func}({fname})"
                func_name = func
                base = fname
            terms[key] = {
                "key": key,
                "base": base,
                "func": func_name,
                "coef_surv": coef_surv,
                "coef_dens": coef_dens,
                "mu": float(mu[i]),
                "sigma": float(sigma[i]),
                "flat_index": idx
            }
    bias_surv = float(b[0]); bias_dens = float(b[1])
    return terms, bias_surv, bias_dens

def evaluate_simplified_formula(model, scaler, ds_val, feature_names, funcs, top_k=None, threshold=None, device="cpu"):
    """
    使用简化后的公式（top_k 或 threshold）重新计算验证集。
    自动识别 fidx 类型（int/str/tuple/函数名），安全评估。
    """
    model.eval()
    model = model.to(device)
    X = torch.from_numpy(ds_val.X).to(device)
    with torch.no_grad():
        ps_full, pd_full, _, _ = model(X)
    ps_full = ps_full.cpu().numpy()
    pd_full = pd_full.cpu().numpy()
    ys = ds_val.y_surv
    yd = ds_val.y_dens
    terms, bias_s, bias_d = extract_weight_terms(model, scaler, feature_names, funcs)
    surv_info = make_readable_formula(terms, bias_s, which="surv", top_k=top_k, threshold=threshold)
    dens_info = make_readable_formula(terms, bias_d, which="dens", top_k=top_k, threshold=threshold)

    surv_terms = surv_info["terms"]
    dens_terms = dens_info["terms"]
    Xn = ds_val.X 
    def eval_terms_from_parts(Xn, parts, funcs_list):
        """
        Xn: numpy standardized feature array shape (N, n_features)
        parts: list of tuples (coef, expr, key, flat_index) returned by make_readable_formula
        funcs_list: list of func names in same order as used to create flat_index (e.g. NONLINEAR_FUNCS)
        """
        func_map = {
            "x": lambda z: z,
            "x^2": lambda z: z**2,
            "log": lambda z: np.log(np.clip(z + 1e-6, 1e-12, None)),
            "exp_neg": lambda z: np.exp(-z),
            "tanh": lambda z: np.tanh(z)
        }
        n_features = Xn.shape[1]
        logit = np.zeros(Xn.shape[0], dtype=float)
        for coef, expr, key, flat_idx in parts:
            func_idx = flat_idx // n_features  
            feat_idx = flat_idx % n_features
            func_name = funcs_list[func_idx]
            if func_name == "exp_neg":
                f = func_map["exp_neg"]
            elif func_name == "x^2":
                f = func_map["x^2"]
            elif func_name in func_map:
                f = func_map[func_name]
            else:
                f = func_map["x"]  
            vals = f(Xn[:, feat_idx])
            logit += coef * vals
        return logit
    logit_s = eval_terms_from_parts(Xn, surv_terms, funcs)
    logit_d = eval_terms_from_parts(Xn, dens_terms, funcs)
    ps_simpl = 1 / (1 + np.exp(-logit_s))
    pd_simpl = 1 / (1 + np.exp(-logit_d))

    auc_full = roc_auc_score((ys > 0.5).astype(int), ps_full)
    auc_simpl = roc_auc_score((ys > 0.5).astype(int), ps_simpl)
    mse_full = mean_squared_error(yd, pd_full)
    mse_simpl = mean_squared_error(yd, pd_simpl)
    # ===============================
    # 🔍 简化公式性能比较（R² 与 MSE）
    # ===============================
    y_true = ys                      # 真实值
    y_pred_full = ps_full            # 完整模型预测
    y_pred_simplified = ps_simpl     # 简化公式预测
    r2_full = r2_score(y_true, y_pred_full)
    r2_simplified = r2_score(y_true, y_pred_simplified)
    mse_full = mean_squared_error(y_true, y_pred_full)
    mse_simplified = mean_squared_error(y_true, y_pred_simplified)
    print("===============================")
    print("🔍 简化公式性能比较")
    print(f"完整模型: R²={r2_full:.4f}, MSE={mse_full:.6f}")
    print(f"简化模型: R²={r2_simplified:.4f}, MSE={mse_simplified:.6f}")
    print(f"⚖️ R² 保留率: {r2_simplified / r2_full * 100:.2f}%")
    print(f"⚖️ MSE 误差比例: {mse_simplified / mse_full:.2f}x")
    print("===============================")
    return {
        "auc_full": auc_full, "auc_simpl": auc_simpl,
        "mse_full": mse_full, "mse_simpl": mse_simpl
    }
def make_readable_formula(terms_dict, bias, which="surv", top_k=None, threshold=None):
    """
    返回结构化结果：{'mode':..., 'formula':..., 'terms': [(coef, expr, key, flat_index), ...]}
    每一项 expr 已经是字符串，且基于标准化表达 (var - mu)/sigma。
    """
    rows = []
    for k, v in terms_dict.items():
        coef = v["coef_surv"] if which == "surv" else v["coef_dens"]
        rows.append((k, abs(coef), coef, v["mu"], v["sigma"], v["func"], v["base"], v["flat_index"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    if len(rows) == 0:
        return {"mode": "", "formula": f"{which} model has no terms.", "terms": []}

    max_val = rows[0][1]
    if top_k is not None:
        chosen = rows[:top_k]
        mode = f"top_k={top_k}"
    elif threshold is not None:
        chosen = [r for r in rows if (r[1] / max_val) >= threshold]
        mode = f"threshold={threshold}"
    else:
        chosen = rows 
        mode = "all"

    parts = []
    for name, abscoef, coef, mu, sigma, func, base, flat_idx in chosen:
        var_std = f"(({base}-{mu:.4g})/{sigma:.4g})"
        if func == "x^2":
            expr = f"({var_std})**2"
        elif func == "log":
            expr = f"log({var_std}+1e-6)"
        elif func == "exp_neg":
            expr = f"exp(-{var_std})"
        elif func == "tanh":
            expr = f"tanh({var_std})"
        else:  # 'x' or fallback
            expr = var_std
        parts.append((coef, expr, name, flat_idx))
    term_strs = [f"{coef:+.6g}*{expr}" for coef, expr, _, _ in parts]
    formula = f"logit = {bias:+.6g} " + " ".join(term_strs)
    return {"mode": mode, "formula": formula, "terms": parts}
# ----------------------------
# Printing helpful outputs
# ----------------------------
def print_full_and_simplified(model, scaler, feature_names, funcs, top_k=None, threshold=None):
    terms, bias_surv, bias_dens = extract_weight_terms(model, scaler, feature_names, funcs)
    print("\n=== Extracted terms (coef_surv, coef_dens) ===")
    for k, v in terms.items():
        print(f"{k:20s}  surv={v['coef_surv']:+.6g}  dens={v['coef_dens']:+.6g}")
    print("\nBiases: surv_bias={:+.6g} , dens_bias={:+.6g}".format(bias_surv, bias_dens))

    print("\n--- Simplified Survival Formula ---")
    surv_info = make_readable_formula(terms, bias_surv, which="surv", top_k=top_k, threshold=threshold)
    print(f"模式: {surv_info['mode']}\n{surv_info['formula']}\nP_surv = sigmoid(logit)\n")

    print("\n--- Simplified Density Formula ---")
    dens_info = make_readable_formula(terms, bias_dens, which="dens", top_k=top_k, threshold=threshold)
    print(f"模式: {dens_info['mode']}\n{dens_info['formula']}\nP_dens = sigmoid(logit)\n")

    return surv_info, dens_info
# ----------------------------
# Plot helpers
# ----------------------------
def plot_fit(out_df):
    plt.figure(figsize=(5,4))
    plt.scatter(out_df["surv_obs"], out_df["surv_pred"], alpha=0.7)
    plt.plot([0,1],[0,1],"r--")
    plt.xlabel("Observed Survival")
    plt.ylabel("Predicted Survival")
    plt.title("Survival Fit")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(5,4))
    plt.scatter(out_df["dens_obs"], out_df["dens_pred"], alpha=0.7)
    plt.plot([0,1],[0,1],"r--")
    plt.xlabel("Observed Density")
    plt.ylabel("Predicted Density")
    plt.title("Density Fit")
    plt.tight_layout(); plt.show()

def plot_diagnostics(df_val, out):
    print("Survival counts (obs):", np.bincount((out['surv_obs']>0.5).astype(int)))
    print("Survival counts (pred):", np.bincount((out['surv_pred']>0.5).astype(int)))
    print("Confusion matrix (obs vs pred):")
    print(confusion_matrix((out['surv_obs']>0.5).astype(int), (out['surv_pred']>0.5).astype(int)))
    plt.figure(); plt.hist(df_val["p0"].values, bins=10); plt.title("p0 distribution"); plt.show()
    mask = out["dens_obs"] > 0
    if mask.sum() > 0:
        resids = out.loc[mask, "dens_pred"] - out.loc[mask, "dens_obs"]
        plt.figure(); plt.hist(resids.values, bins=12); plt.title("density residuals (survived only)"); plt.show()
# ----------------------------
# Main
# ----------------------------
def main():
    # ==============================================================
    # 🧩 用户自定义配置区 —— 直接在此修改
    # ==============================================================

    USE_GPU = True             # 是否使用 GPU（如果可用）
    EPOCHS = 400               # 训练轮数
    LEARNING_RATE = 1e-2       # 学习率
    BATCH_SIZE = 32            # 批大小
    RESULTS_ROOT = "./results" # 数据文件夹路径

    # 二选一（只需修改一个）
    TOP_K = 13                 # 保留前多少项（设为 None 可禁用）
    THRESHOLD = None             # 按权重比例筛选（如 0.15），设为 None 可禁用

    # ==============================================================

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"💻 使用设备: {device}")

    print("📂 载入所有 group 数据...")
    df = load_all_results(RESULTS_ROOT)
    print(f"✅ 数据量: {len(df)} 行")

    # 数据清理与分割
    df = df.dropna(subset=["survival_prob"]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42)  # 打乱
    n_train = int(0.8 * len(df))
    df_train, df_val = df.iloc[:n_train], df.iloc[n_train:]

    # 标准化
    scaler = StandardScaler().fit(df_train[FEATURE_NAMES])
    ds_train, ds_val = SimDataset(df_train, scaler), SimDataset(df_val, scaler)

    # 模型初始化
    model = ExplainableNonlinearModel(
        n_features=len(FEATURE_NAMES),
        funcs=NONLINEAR_FUNCS
    )

    # 训练模型
    print(f"🚀 开始训练 (epochs={EPOCHS}, lr={LEARNING_RATE})...")
    train_model(
        model, ds_train,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        device=device,
        lambda_density=1.0
    )

    # 评估
    print("\n📊 模型评估中...")
    res_val = evaluate_model(model, ds_val, device=device)
    print(f"📈 验证集 R² (Survival) = {res_val['r2_surv']:.4f}")
    print(f"📉 验证集 R² (Density)  = {res_val['r2_dens']:.4f}")
    print(f"📈 验证集 MSE (Survival) = {res_val['mse_surv']:.6f}")
    print(f"📉 验证集 MSE (Density)  = {res_val['mse_dens']:.6f}")

    # 打印并简化公式
    print("\n🧮 提取与简化公式:")
    print_full_and_simplified(
        model, scaler,
        FEATURE_NAMES, NONLINEAR_FUNCS,
        top_k=TOP_K, threshold=THRESHOLD
    )

    # 保存预测结果
    out = pd.DataFrame({
        "p0": df_val["p0"],
        "surv_obs": df_val["survival_prob"],
        "surv_pred": res_val["ps"],
        "dens_obs": df_val["stable_density_mean"],
        "dens_pred": res_val["pd"],
    })
    out.to_csv("nonlinear_sigmoid_fit_results.csv", index=False)
    print("💾 已保存预测结果到 nonlinear_sigmoid_fit_results.csv")
    import datetime
    # 生成可读公式文本
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# 非线性拟合模型结果汇总\n")
    lines.append(f"**生成时间：** {timestamp}\n")
    lines.append("## 🧠 训练配置\n")
    lines.append(f"- 设备: {device}")
    lines.append(f"- 训练轮数: {EPOCHS}")
    lines.append(f"- 学习率: {LEARNING_RATE}")
    lines.append(f"- 批大小: {BATCH_SIZE}")
    lines.append("")

    lines.append("## 📊 模型性能\n")
    lines.append(f"- 验证集 R² (Survival): {res_val['r2_surv']:.4f}")
    lines.append(f"- 验证集 R² (Density): {res_val['r2_dens']:.4f}")
    lines.append(f"- 验证集 MSE (Survival): {res_val['mse_surv']:.6f}")
    lines.append(f"- 验证集 MSE (Density): {res_val['mse_dens']:.6f}")

    lines.append("## 🧮 简化公式\n")
    lines.append(f"模式: {'top_k='+str(TOP_K) if TOP_K else 'threshold='+str(THRESHOLD)}\n")

    # 重新提取并写入公式文本
    terms, bias_surv, bias_dens = extract_weight_terms(model, scaler, FEATURE_NAMES, NONLINEAR_FUNCS)
    surv_info = make_readable_formula(terms, bias_surv, which="surv", top_k=TOP_K, threshold=THRESHOLD)
    dens_info = make_readable_formula(terms, bias_dens, which="dens", top_k=TOP_K, threshold=THRESHOLD)

    lines.append("### Survival 模型公式\n")
    lines.append("```\n" + surv_info["formula"] + "\nP_survival = sigmoid(logit)\n```\n")

    lines.append("### Density 模型公式\n")
    lines.append("```\n" + dens_info["formula"] + "\nP_density = sigmoid(logit)\n```\n")
    result_perf = evaluate_simplified_formula(model, scaler, ds_val, FEATURE_NAMES, NONLINEAR_FUNCS, top_k=TOP_K, threshold=THRESHOLD)
    lines.append("## 🔍 简化公式性能比较\n")
    lines.append(f"- 验证集 R² (Survival): {res_val['r2_surv']:.4f}")
    lines.append(f"- 验证集 R² (Density): {res_val['r2_dens']:.4f}")
    lines.append(f"- 验证集 MSE (Survival): {res_val['mse_surv']:.6f}")
    lines.append(f"- 验证集 MSE (Density): {res_val['mse_dens']:.6f}")
    lines.append("")
    # 输出文件
    save_path = "fit_formula_summary.md"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"🧾 已保存公式至 {save_path}")

    # 绘制拟合效果图
    print("\n📈 绘制拟合与诊断图...")
    plot_fit(out)
    plot_diagnostics(df_val, out)
    evaluate_simplified_formula(
        model, scaler, ds_val,
        FEATURE_NAMES, NONLINEAR_FUNCS,
        top_k=TOP_K, threshold=THRESHOLD,
        device=device
        )
    print("\n✅ 全部完成。")


if __name__ == "__main__":
    main()
