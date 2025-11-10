import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import multiprocessing as mp
from functools import partial
import time
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.stats import sem, t
import matplotlib.animation as animation
import json
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
mpl.rcParams['mathtext.fontset'] = 'cm' 
# --- 全局绘图风格 ---
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'lines.linewidth': 1.6,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

sns.set_palette("crest") 
# 字体配置
font_path = r"C:\Windows\Fonts\msyh.ttc"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = 'Microsoft YaHei'
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# ------------------- 全局绘图风格 -------------------
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['lines.linewidth'] = 1.6
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.unicode_minus'] = False
# 尝试中文字体配置（如果有）
try:
    if os.name == 'nt':  # Windows
        if fm.findfont('Microsoft YaHei'):
            mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        elif fm.findfont('SimHei'):
            mpl.rcParams['font.sans-serif'] = ['SimHei']
except Exception:
    pass
# ------------------- 模块 1: 元胞自动机核心 -------------------
class CellularAutomaton:
    """
    元胞自动机：
    - N x N 网格
    - 每步：死亡判定（拥挤相关） -> 活细胞尝试繁殖 -> 空格判断是否出生（受邻居繁殖尝试数量与阈值约束）
    """
    def __init__(self, N=100, initial_density=0.3, birth=0.2, base_death=0.5, alpha=2.0, N_birth_max=4, N_death_min=4):
        self.N = N
        self.P_birth = birth
        self.P_death_base = base_death
        self.alpha = alpha
        self.N_birth_max = N_birth_max
        self.N_death_min = N_death_min
        self.initial_density = initial_density
        self.grid = (np.random.rand(N, N) < initial_density).astype(np.int8)

    def get_population(self):
        return int(self.grid.sum())

    def get_density(self):
        return float(self.get_population() / (self.N * self.N))

    def _neighbor_count(self, arr=None):
        if arr is None:
            arr = self.grid
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        return convolve2d(arr, kernel, mode='same', boundary='wrap')
    def evolve_one_step(self):
        """
        同步更新（改进版）：
        1) 先对所有活细胞判定死亡并立即应用；
        2) 在新网格上，让存活的活细胞尝试繁殖；
        3) 对空格：若周围有繁殖尝试且数量 < N_birth_max，则出生。
        """
        N = self.N
        grid = self.grid
        neighbor_count = self._neighbor_count(grid)
        # ------------------ 死亡阶段 ------------------
        is_alive = (grid == 1)
        is_empty = ~is_alive
        # 基础死亡概率
        P_death = np.zeros_like(grid, dtype=float)
        P_death[is_alive] = self.P_death_base
        # 拥挤惩罚（仅当邻居数 ≥ N_death_min 时）
        crowded = is_alive & (neighbor_count >= self.N_death_min)
        P_prime = self.P_death_base * (1 + self.alpha * (neighbor_count / 8.0))
        P_prime = np.clip(P_prime, 0, 1)
        P_death[crowded] = P_prime[crowded]
        # 死亡判定
        rand_death = np.random.rand(N, N)
        survive_mask = (rand_death >= P_death)
        grid = grid * survive_mask  # 死亡的细胞立即清除
        # ------------------ 繁殖阶段 ------------------
        is_alive = (grid == 1)
        is_empty = (grid == 0)
        # 活细胞尝试繁殖
        rand_birth_attempt = np.random.rand(N, N)
        attempted_to_reproduce = is_alive & (rand_birth_attempt < self.P_birth)
        # 计算每个空格周围的繁殖者数量
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        repro_neighbors = convolve2d(attempted_to_reproduce.astype(int), kernel, mode='same', boundary='wrap')
        # 出生条件：至少一个繁殖者，且少于阈值
        has_repro = (repro_neighbors > 0)
        birth_forbidden = (repro_neighbors >= self.N_birth_max)
        births = is_empty & has_repro & (~birth_forbidden)
        # ------------------ 更新网格 ------------------
        new_grid = np.zeros_like(grid, dtype=np.int8)
        new_grid[is_alive | births] = 1
        self.grid = new_grid
# ------------------- 顶层单次运行函数（用于 multiprocessing） -------------------
def single_run(seed, N, p0, P_birth, P_death_base, alpha, N_birth_max, N_death_min, T_max):
    """
    顶层单次运行函数，必须是模块顶层以便可被 multiprocessing pickle。
    返回 dict 格式：{'p0', 'final_density', 'time_series', 'final_grid', 'seed'}
    """
    np.random.seed(int(seed) & 0x7fffffff)
    CA = CellularAutomaton(
        N=N,
        initial_density=p0,
        birth=P_birth,
        base_death=P_death_base,
        alpha=alpha,
        N_birth_max=N_birth_max,
        N_death_min=N_death_min
    )
    ts = [CA.get_density()]
    for _ in range(T_max):
        CA.evolve_one_step()
        ts.append(CA.get_density())
    return {
        'p0': float(p0),
        'final_density': float(ts[-1]),
        'time_series': ts,  # list of length T_max+1
        'final_grid': CA.grid.copy(),
        'seed': int(seed)
    }
# ------------------- 模块 2: MonteCarloStudy -------------------
class MonteCarloStudy:
    def __init__(self, N=100, P_birth=0.2, P_death_base=0.5, alpha=4.0,
                 N_birth_max=3, N_death_min=5, p0_values=None, R=500, T_max=500, n_workers=None):
        self.N = N
        self.P_birth = float(P_birth)
        self.P_death_base = float(P_death_base)
        self.alpha = float(alpha)
        self.N_birth_max = int(N_birth_max)
        self.N_death_min = int(N_death_min)
        if p0_values is None:
            p0_values = np.concatenate([np.linspace(0.001, 0.05, 20), np.linspace(0.06, 0.9, 10)]).round(4)
        self.p0_values = np.array(p0_values)
        self.R = int(R)
        self.T_max = int(T_max)
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        # 存放结构
        self.all_density_data = []   
        self.time_series_data = {}  
        self.df = None
        self.performance_log = None
        self.performance_df = None
        self.survival_results = None
        self.stable_density_results = None
        self.cluster_summary = None
    def run_all_experiments(self, force_rerun=False):
        """
        并行运行所有 p0 的蒙特卡洛试验。
        如果 montecarlo_results.csv 已存在且 force_rerun=False，则会跳过计算并加载 CSV。
        """
        # 如果已有文件并且不强制重跑，则直接加载并构建内部对象
        if (not force_rerun) and os.path.exists("montecarlo_results.csv") and os.path.exists("performance_report.csv"):
            print("检测到已有 montecarlo_results.csv 与 performance_report.csv，将直接加载（如需重跑请设置 force_rerun=True）")
            self.df = pd.read_csv("montecarlo_results.csv", converters={'final_grid': lambda x: x})
            # 尝试加载 performance
            try:
                self.performance_df = pd.read_csv("performance_report.csv")
            except Exception:
                self.performance_df = None
            # time_series_data 需要重建
            self._rebuild_time_series_from_df()
            return self.df

        # 否则按参数并行运行
        start_all = time.time()
        print(f"--- 概率元胞自动机蒙特卡洛研究 (并行加速版) ---")
        print(f"网格: {self.N}x{self.N} | R: {self.R} 次 | T_max: {self.T_max} 步")
        print(f"P_birth: {self.P_birth:.4f} | P_death_base: {self.P_death_base:.4f} | Alpha: {self.alpha}")
        print(f"N_birth_max: {self.N_birth_max} | N_death_min: {self.N_death_min}")
        print(f"CPU 并行核心数: {self.n_workers}")
        print("------------------------------------------------------------\n")

        self.all_density_data = []
        self.time_series_data = {}
        perf_log = []

        global_run_counter = 0
        for p0 in self.p0_values:
            print(f"> 正在运行初始密度 p0 = {p0:.4f} 的实验...")
            t0 = time.time()
            seeds = np.random.randint(0, 2**31 - 1, size=self.R)
            func = partial(single_run,
                           N=self.N,
                           p0=p0,
                           P_birth=self.P_birth,
                           P_death_base=self.P_death_base,
                           alpha=self.alpha,
                           N_birth_max=self.N_birth_max,
                           N_death_min=self.N_death_min,
                           T_max=self.T_max)
            with mp.Pool(processes=self.n_workers) as pool:
                results = pool.map(func, seeds)
            # 为每个结果分配全局唯一 run_id，并把需要保存的键转为可序列化（final_grid 也可序列化为 list 或 bytes）
            for local_idx, res in enumerate(results):
                run_id = global_run_counter
                res['run_id'] = int(run_id)
                # 为防止 pandas 无法直接存储 ndarray，final_grid 序列化为 bytes via tostring 或 list
                res['final_grid'] = res['final_grid'].astype(np.int8).tolist()
                self.all_density_data.append(res)
                global_run_counter += 1
            # 平均时间序列
            avg_series = np.mean([r['time_series'] for r in results], axis=0)
            self.time_series_data[float(p0)] = avg_series

            t1 = time.time()
            elapsed = t1 - t0
            per_run = elapsed / self.R
            perf_log.append({'p0': float(p0), 'runs': self.R, 'total_time_s': round(elapsed, 4), 'time_per_run_s': round(per_run, 6)})
            print(f"  - p0={p0:.4f}: 完成 {self.R} 次运行，用时 {elapsed:.2f} 秒，平均每次 {per_run:.4f} 秒。")
        total_elapsed = time.time() - start_all
        print(f"\n✅ 所有模拟实验完成！总耗时: {total_elapsed:.2f} 秒")
        self.df = pd.DataFrame(self.all_density_data)
        if 'final_grid' in self.df.columns:
            self.df['final_grid'] = self.df['final_grid'].apply(lambda g: json.dumps(g))
        self.df.to_csv("montecarlo_results.csv", index=False)
        # 性能报告
        self.performance_log = perf_log
        self.performance_df = pd.DataFrame(perf_log)
        self.performance_df.to_csv("performance_report.csv", index=False)
        print(f"DataFrame 准备完毕，共计 {len(self.df)} 条记录。 performance_report.csv 已保存。")
        return self.df
    def _rebuild_time_series_from_df(self):
        """从 montecarlo_results.csv 中重建 time_series_data（当已保存 CSV 时使用）"""
        if self.df is None:
            self.df = pd.read_csv("montecarlo_results.csv")
        if 'time_series' in self.df.columns:
            # 如果保存过列表形式字符串，需要 eval 或 json.loads
            def parse_ts(x):
                try:
                    return json.loads(x)
                except Exception:
                    try:
                        return eval(x)
                    except Exception:
                        return None
            grouped = {}
            for p0, group in self.df.groupby('p0'):
                series_list = [parse_ts(v) for v in group['time_series'].values if parse_ts(v) is not None]
                if len(series_list) > 0:
                    grouped[float(p0)] = np.mean(series_list, axis=0)
            self.time_series_data = grouped
            return
        else:
            self.time_series_data = {}
            print("注意：montecarlo_results.csv 中无 time_series 字段，无法重建完整 time series。")
    # -------- 分析方法 ----------
    def analyze(self):
        """
        生成：
          - self.survival_results (DataFrame: p0, survival_prob, CI_lower, CI_upper)
          - self.stable_density_results (DataFrame: p0, stable_density_mean, stable_density_se, margin_of_error)
        返回 time_series_data（字典 p0 -> series）
        """
        if self.df is None:
            raise RuntimeError("self.df 为空，请先运行 run_all_experiments() 或加载 montecarlo_results.csv")
        def parse_final_grid(x):
            if isinstance(x, str):
                try:
                    arr = np.array(json.loads(x), dtype=np.int8)
                    return arr
                except Exception:
                    return None
            elif isinstance(x, list):
                return np.array(x, dtype=np.int8)
            else:
                return x

        self.df['final_grid_parsed'] = self.df['final_grid'].apply(parse_final_grid) if 'final_grid' in self.df.columns else None
        # 1) 生存概率：以 final_density > 0 判为存活
        summary = self.df.groupby(['p0', 'run_id'])['final_density'].last().reset_index()
        survival = summary.groupby('p0')['final_density'].agg(lambda x: (x > 0).sum()).reset_index(name='survived_runs')
        survival['total_runs'] = self.R
        survival['survival_prob'] = survival['survived_runs'] / survival['total_runs']
        survival['se'] = np.sqrt((survival['survival_prob'] * (1 - survival['survival_prob'])) / survival['total_runs'])
        z = 1.96
        survival['CI_lower'] = (survival['survival_prob'] - z * survival['se']).clip(0, 1)
        survival['CI_upper'] = (survival['survival_prob'] + z * survival['se']).clip(0, 1)
        self.survival_results = survival[['p0', 'survival_prob', 'CI_lower', 'CI_upper']].sort_values('p0').reset_index(drop=True)
        # 2) 稳定密度：仅对 final_density>0 的样本计算均值与 95% CI（t 分布）
        survived_df = self.df[self.df['final_density'] > 0]
        stable_rows = []
        for p0, group in survived_df.groupby('p0'):
            vals = group['final_density'].astype(float).values
            n = len(vals)
            if n == 0:
                mean = 0.0; semv = 0.0; moe = 0.0
            else:
                mean = np.mean(vals)
                semv = sem(vals) if n > 1 else 0.0
                moe = semv * t.ppf(0.975, df=max(1, n - 1)) if n > 1 else 0.0
            stable_rows.append({
                'p0': float(p0),
                'stable_density_mean': mean,
                'stable_density_se': semv,
                'margin_of_error': moe
            })
        stable_df = pd.DataFrame(stable_rows)
        if stable_df.empty or 'p0' not in stable_df.columns:
            print("⚠️ 无存活样本，无法计算稳态密度。")
            stable_df = pd.DataFrame(columns=['p0', 'stable_density_mean', 'stable_density_se', 'margin_of_error'])
        else:
            stable_df = stable_df.sort_values('p0').reset_index(drop=True)

        self.stable_density_results = stable_df

        # 打印摘要
        print("\n--- 分析完毕：存活概率 & 稳态密度已生成 ---")
        try:
            print(self.survival_results.to_markdown(index=False, floatfmt=".4f"))
            if not self.stable_density_results.empty:
                print(self.stable_density_results.head().to_markdown(index=False, floatfmt=".4f"))
            else:
                print("（所有样本灭绝，无稳态密度数据。）")
        except Exception:
            print(self.survival_results.head())
            print(self.stable_density_results.head())

        # 保存 summary
        self.survival_results.to_csv("survival_summary.csv", index=False)
        self.stable_density_results.to_csv("density_summary.csv", index=False)
        return self.time_series_data


    # --------- 集群统计 ----------
    def _calculate_cluster_stats(self, save_csv=True):
        """
        对存活样本（final_density>0）计算集群统计：
        - mean_N_clusters, mean_S_avg, mean_S_max, se_S_avg, se_S_max
        结果保存在 self.cluster_summary
        """
        print("\n--- III. 集群几何分析 ---")
        if self.df is None or self.df.empty:
            print("⚠️ 无可用数据 (self.df 为空)，无法计算集群统计。")
            self.cluster_summary = pd.DataFrame()
            return self.cluster_summary

        # 使用解析后的 final_grid（如果存在）
        if 'final_grid_parsed' not in self.df or self.df['final_grid_parsed'].isnull().all():
            print("⚠️ 未能找到 final_grid 数据（用于集群统计）。如需集群统计，请在 run_all_experiments 中保留 final_grid。")
            self.cluster_summary = pd.DataFrame()
            return self.cluster_summary

        df_survived = self.df[self.df['final_density'] > 0].copy()
        stats_list = []
        struct = ndimage.generate_binary_structure(2, 2) 

        for idx, row in df_survived.iterrows():
            fg = row['final_grid_parsed']
            if fg is None:
                stats_list.append({'p0': row['p0'], 'run_id': row['run_id'], 'N_clusters': 0, 'Avg_cluster_size': 0.0, 'Max_cluster_size': 0.0})
                continue
            labeled, nfeat = ndimage.label(fg, structure=struct)
            if nfeat == 0:
                stats_list.append({'p0': row['p0'], 'run_id': row['run_id'], 'N_clusters': 0, 'Avg_cluster_size': 0.0, 'Max_cluster_size': 0.0})
                continue
            sizes = ndimage.sum(fg, labeled, range(1, nfeat+1))
            stats_list.append({'p0': row['p0'], 'run_id': row['run_id'], 'N_clusters': int(nfeat),
                               'Avg_cluster_size': float(np.mean(sizes)), 'Max_cluster_size': float(np.max(sizes))})
        df_clusters = pd.DataFrame(stats_list)
        if df_clusters.empty:
            self.cluster_summary = pd.DataFrame()
            return self.cluster_summary

        summary = df_clusters.groupby('p0').agg(
            mean_N_clusters=('N_clusters', 'mean'),
            mean_S_avg=('Avg_cluster_size', 'mean'),
            mean_S_max=('Max_cluster_size', 'mean'),
            se_S_avg=('Avg_cluster_size', sem),
            se_S_max=('Max_cluster_size', sem)
        ).reset_index().round(4)
        self.cluster_summary = summary
        if save_csv:
            summary.to_csv("cluster_summary.csv", index=False)
        print("集群统计完成。")
        return summary

# ------------------- 模块 3: 绘图函数 -------------------
def plot_time_series(time_series_data, N_birth_max, N_death_min, P_birth, P_death_base, alpha, save_prefix="Figure_time_series"):
    if time_series_data is None or len(time_series_data) == 0:
        print("⚠️ time_series_data 为空，无法绘图。")
        return
    keys = sorted(time_series_data.keys())
    series_matrix = np.vstack([time_series_data[k] for k in keys]).T  
    plt.figure(figsize=(8,6))  
    colors = plt.cm.viridis(np.linspace(0, 1, len(keys)))

    for i, p0 in enumerate(keys):
        plt.plot(series_matrix[:, i], color=colors[i], lw=1.3, label=fr"$\rho_0={p0:.3f}$")

    plt.xlabel("时间步 $t$")
    plt.ylabel(r"平均密度 $\bar{\rho}(t)$")
    plt.title(fr"平均密度收敛（$P_b={P_birth}, P_{{db}}={P_death_base}, \alpha={alpha}, N_b={N_birth_max}, N_d={N_death_min}$）")
    
    # --- 图例优化部分 ---
    ncol = 6 if len(keys) > 20 else 4  
    plt.legend(
        ncol=ncol,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12), 
        fontsize=7.2,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.2,
        handletextpad=0.4
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    png = f"{save_prefix}.png"
    pdf = f"{save_prefix}.pdf"
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"保存：{png}, {pdf}")

def plot_survival_probability(df_surv, N_birth_max, N_death_min, P_birth, P_death_base, alpha, save_prefix="Figure_survival"):
    if df_surv is None or df_surv.empty:
        print("⚠️ survival_results 为空，无法绘图。")
        return
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(df_surv['p0'], df_surv['survival_prob'], 'o-', color='royalblue', label=r'$\hat{P}_{survive}$')
    plt.fill_between(df_surv['p0'], df_surv['CI_lower'], df_surv['CI_upper'], color='royalblue', alpha=0.18)
    critical = df_surv.loc[df_surv['survival_prob'] < 1, 'p0']
    if not critical.empty:
        cp = critical.min()
        plt.axvline(cp, color='orangered', linestyle='--', linewidth=1.6, label=fr'$\rho_c \approx {cp:.4f}$')
    plt.ylim(max(0.0, df_surv['survival_prob'].min()-0.02), 1.02)
    plt.xlabel(r"$\rho_0$")
    plt.ylabel(r"存活概率 $\hat{P}_{survive}$")
    plt.title(fr"存活概率 vs 初始密度（$P_b={P_birth}, P_{{db}}={P_death_base}, \alpha={alpha}$）")
    plt.legend(frameon=False, loc='lower right')
    png = f"{save_prefix}.png"
    pdf = f"{save_prefix}.pdf"
    plt.savefig(png, dpi=600, bbox_inches='tight')
    plt.savefig(pdf, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"保存：{png}, {pdf}")

def plot_stable_density(df_stable, N_birth_max, N_death_min, P_birth, P_death_base, alpha, save_prefix="Figure_stable_density"):
    if df_stable is None or df_stable.empty:
        print("⚠️ stable_density_results 为空，无法绘图。")
        return
    plt.figure(figsize=(6.5, 4.2))
    plt.errorbar(
        df_stable['p0'],
        df_stable['stable_density_mean'],
        yerr=df_stable['margin_of_error'],
        fmt='o', markersize=4,  
        capsize=2.5,            
        color='forestgreen',
        ecolor='darkgreen',
        elinewidth=1.0,         
        alpha=0.85              
    )
    ymin = df_stable['stable_density_mean'].min() - 0.01
    ymax = df_stable['stable_density_mean'].max() + 0.01
    plt.ylim(ymin, ymax)
    plt.xlabel(r"$\rho_0$")
    plt.ylabel(r"稳态密度 $\bar{\rho}_{stable}$")
    plt.title(fr"稳态密度 vs 初始密度（$P_b={P_birth}, P_{{db}}={P_death_base}, \alpha={alpha}$）")
    png = f"{save_prefix}.png"
    pdf = f"{save_prefix}.pdf"
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"保存：{png}, {pdf}")

def plot_cluster_stats(df_cluster, N_birth_max, N_death_min, P_birth, P_death_base, alpha, save_prefix="Figure_cluster"):
    if df_cluster is None or df_cluster.empty:
        print("⚠️ cluster_summary 为空或未计算。")
        return
    plt.figure(figsize=(6.5, 4.2))
    plt.errorbar(
    df_cluster['p0'], df_cluster['mean_S_avg'], yerr=df_cluster['se_S_avg'],
    fmt='o', markersize=5,
    color='purple', mfc='purple', mec='white',
    ecolor='gray', elinewidth=0.8, capsize=1.8, capthick=0.8,
    alpha=0.9
    )
    plt.xlabel(r"$\rho_0$")
    plt.ylabel("平均集群大小")
    plt.title(fr"平均集群大小 vs 初始密度（仅存活样本）")
    png = f"{save_prefix}.png"
    pdf = f"{save_prefix}.pdf"
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"保存：{png}, {pdf}")

# ------------------- 主程序 -------------------
if __name__ == "__main__":
    # Windows-safe multiprocess start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # ---------- 参数（如需调整请在这里改） ----------
    N_GRID = 100
    R_RUNS = 200
    T_MAX_STEPS = 100
    P_BIRTH = 0.7
    P_DEATH_BASE = 0.3
    ALPHA = 2.0
    N_BIRTH_MAX = 6
    N_DEATH_MIN = 3
    FORCE_RERUN = True  # True 强制重跑并覆盖已有 CSV
    #可选p0 values 预设（稠密采样低密度）
    #P0_VALUES = np.concatenate([np.array([0.0005, 0.001, 0.002, 0.005, 0.01]),np.linspace(0.02, 0.9, 30)]).round(4)
    P0_VALUES=np.linspace(0.05, 0.95, 15).round(4)
    study = MonteCarloStudy(
        N=N_GRID,
        P_birth=P_BIRTH,
        P_death_base=P_DEATH_BASE,
        alpha=ALPHA,
        N_birth_max=N_BIRTH_MAX,
        N_death_min=N_DEATH_MIN,
        p0_values=P0_VALUES,
        R=R_RUNS,
        T_max=T_MAX_STEPS,
        n_workers=max(1, mp.cpu_count() - 1)
    )

    # 运行（若已有结果且不强制 rerun，则直接加载）
    df = study.run_all_experiments(force_rerun=FORCE_RERUN)

    # 分析并保存 summary（生存概率、稳态密度）
    time_series_data = study.analyze()

    # 绘图
    plot_time_series(study.time_series_data, N_BIRTH_MAX, N_DEATH_MIN, P_BIRTH, P_DEATH_BASE, ALPHA)
    if study.survival_results is not None:
        plot_survival_probability(study.survival_results, N_BIRTH_MAX, N_DEATH_MIN, P_BIRTH, P_DEATH_BASE, ALPHA)
    if study.stable_density_results is not None:
        plot_stable_density(study.stable_density_results, N_BIRTH_MAX, N_DEATH_MIN, P_BIRTH, P_DEATH_BASE, ALPHA)

    # 集群统计与绘图（如果需要）
    cluster_summary = study._calculate_cluster_stats(save_csv=True)
    if cluster_summary is not None and not cluster_summary.empty:
        plot_cluster_stats(cluster_summary, N_BIRTH_MAX, N_DEATH_MIN, P_BIRTH, P_DEATH_BASE, ALPHA)

    print("\n全部流程完成。结果和图像保存在当前目录。")
