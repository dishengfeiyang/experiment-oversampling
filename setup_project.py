# setup_project.py
"""
一键创建项目目录结构和所有代码文件。
在项目根目录下运行：
    python setup_project.py
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

FILES = {

# ==================== requirements.txt ====================
"requirements.txt": """numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
pyyaml>=6.0
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
joblib>=1.3
""",

# ==================== configs/default.yaml ====================
"configs/default.yaml": """data:
  n: 100000
  k_n: 500
  d: 10
  mu_scale: 1.0

synthesizer:
  type: gaussian
  D: 1.0

classifier:
  type: logistic
  C: 1.0
  max_iter: 1000

allocation:
  w0: 0.5
  w1: 0.5
  c_n: 2.1

experiment:
  n_repeats: 100
  seed_base: 42
  test_size: 5000
  val_size: 2000
  fidelity_ratio: 0.1

output:
  results_dir: results/raw
  figures_dir: results/figures
""",

# ==================== configs/experiment_grid.yaml ====================
"configs/experiment_grid.yaml": """beta_grid: [0.3, 0.5, 0.7]
epsilon_grid: [0.0, 0.05, 0.1]

methods:
  - real_only
  - naive_balancing_default
  - naive_balancing_conservative
  - naive_balancing_aggressive
  - naive_balancing_optimal_s
  - theoretical_optimal
  - complete_replacement_ctrl
  - complete_replacement_opt
  - fixed_s_005_optimal_m
  - fixed_s_010_optimal_m
  - fixed_s_050_optimal_m
""",

# ==================== src/utils/seed.py ====================
"src/utils/seed.py": '''"""随机种子设置"""
import random
import numpy as np


def set_seed(seed: int):
    """统一设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
''',

# ==================== src/data/generator.py ====================
"src/data/generator.py": '''"""真实数据生成器"""
import numpy as np
from dataclasses import dataclass


@dataclass
class DataConfig:
    n: int = 100000
    k_n: int = 500
    d: int = 10
    mu_scale: float = 1.0


def generate_gaussian_data(cfg: DataConfig, seed: int):
    """
    生成高斯混合数据：
        P0 = N(0, I)
        P1 = N(mu, I), mu = (mu_scale, 0, ..., 0)
    返回 (X, y, mu)
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros(cfg.d)
    mu[0] = cfg.mu_scale

    X_maj = rng.normal(0, 1, (cfg.n - cfg.k_n, cfg.d))
    X_min = rng.normal(mu, 1, (cfg.k_n, cfg.d))

    X = np.vstack([X_maj, X_min])
    y = np.array([0] * (cfg.n - cfg.k_n) + [1] * cfg.k_n)
    return X, y, mu
''',

# ==================== src/data/split.py ====================
"src/data/split.py": '''"""嵌套划分"""
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class SplitConfig:
    test_size: int = 5000
    val_size: int = 2000
    fidelity_ratio: float = 0.1


@dataclass
class DataSplits:
    X_test: np.ndarray
    y_test: np.ndarray
    X_min_train: np.ndarray
    X_maj_train: np.ndarray
    X_min_fidelity: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    mu_true: np.ndarray = None


def nested_split(X, y, cfg: SplitConfig, seed: int) -> DataSplits:
    """嵌套划分：测试集 / 验证集 / 生成器保真度集 / 分类器训练集"""
    rng = np.random.default_rng(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=cfg.val_size, stratify=y_train, random_state=seed + 1
    )

    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]

    n_fid = max(50, int(len(X_min) * cfg.fidelity_ratio))
    idx = rng.permutation(len(X_min))
    X_min_fid = X_min[idx[:n_fid]]
    X_min_clf = X_min[idx[n_fid:]]

    return DataSplits(
        X_test=X_test, y_test=y_test,
        X_min_train=X_min_clf, X_maj_train=X_maj,
        X_min_fidelity=X_min_fid,
        X_val=X_val, y_val=y_val,
    )
''',

# ==================== src/synthesizer/base.py ====================
"src/synthesizer/base.py": '''"""生成器基类"""
from abc import ABC, abstractmethod
import numpy as np


class BaseSynthesizer(ABC):
    """所有生成器的抽象基类"""

    @abstractmethod
    def fit(self, X_min: np.ndarray, seed: int) -> None:
        pass

    @abstractmethod
    def sample(self, m: int, seed: int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def discrepancy(self) -> float:
        pass
''',

# ==================== src/synthesizer/gaussian.py ====================
"src/synthesizer/gaussian.py": '''"""高斯模拟生成器"""
import numpy as np
from .base import BaseSynthesizer


class GaussianSynthesizer(BaseSynthesizer):
    """
    受控高斯生成器：
        mu_hat = mu_empirical + epsilon * 1 + s^(-beta) * Z
    """

    def __init__(self, mu_true: np.ndarray, beta: float, epsilon: float, D: float = 1.0):
        self.mu_true = mu_true
        self.beta = beta
        self.epsilon = epsilon
        self.D = D
        self.mu_hat = None
        self.s = 0

    def fit(self, X_min: np.ndarray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.s = len(X_min)
        d = X_min.shape[1]

        mu_empirical = X_min.mean(axis=0)
        noise = (self.s ** (-self.beta)) * rng.normal(0, 1, d)
        self.mu_hat = mu_empirical + self.epsilon * np.ones(d) + noise

    def sample(self, m: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 1000)
        return rng.normal(self.mu_hat, 1.0, (m, self.mu_hat.shape[0]))

    @property
    def discrepancy(self) -> float:
        return float(np.linalg.norm(self.mu_hat - self.mu_true))
''',

# ==================== src/synthesizer/registry.py ====================
"src/synthesizer/registry.py": '''"""生成器注册表"""
from .gaussian import GaussianSynthesizer

SYNTHESIZER_REGISTRY = {
    "gaussian": GaussianSynthesizer,
}


def build_synthesizer(name: str, **kwargs):
    if name not in SYNTHESIZER_REGISTRY:
        raise ValueError(f"Unknown synthesizer: {name}")
    return SYNTHESIZER_REGISTRY[name](**kwargs)


def register_synthesizer(name: str, cls):
    SYNTHESIZER_REGISTRY[name] = cls
''',

# ==================== src/allocation/objective.py ====================
"src/allocation/objective.py": '''"""目标函数 J_n(s, m)"""
import numpy as np


def compute_J(s: int, m: int, k_n: int, n: int,
              beta: float, epsilon: float, D: float,
              c_n: float, w0: float = 0.5, w1: float = 0.5) -> float:
    """J_n(s, m) = w1 * (m/N) * Delta(s) + c_n * sqrt(w0^2/(n-k) + w1^2/N)"""
    N = k_n - s + m
    if N <= 0:
        return np.inf
    if s <= 0:
        return np.inf
    delta = D * (s ** (-beta)) + epsilon
    bias = w1 * (m / N) * delta
    sampling = c_n * np.sqrt((w0 ** 2) / (n - k_n) + (w1 ** 2) / N)
    return bias + sampling
''',

# ==================== src/allocation/optimization.py ====================
"src/allocation/optimization.py": '''"""最优分配求解"""
import numpy as np
from .objective import compute_J


def find_optimal_s_fixed_m(k_n, n, m, beta, epsilon, D, c_n,
                            w0=0.5, w1=0.5, s_grid=None):
    if s_grid is None:
        s_grid = np.arange(1, k_n + 1, max(1, k_n // 200))
    best_s, best_J = 1, np.inf
    for s in s_grid:
        J = compute_J(s, m, k_n, n, beta, epsilon, D, c_n, w0, w1)
        if J < best_J:
            best_J = J
            best_s = s
    return int(best_s), float(best_J)


def find_optimal_m_fixed_s(s, k_n, n, beta, epsilon, D, c_n,
                            w0=0.5, w1=0.5, m_grid=None):
    M = n - 2 * k_n + s
    if m_grid is None:
        m_grid = np.linspace(0, M, 100).astype(int)
    best_m, best_J = 0, np.inf
    for m in m_grid:
        J = compute_J(s, m, k_n, n, beta, epsilon, D, c_n, w0, w1)
        if J < best_J:
            best_J = J
            best_m = m
    return int(best_m), float(best_J)


def find_theoretical_optimal(k_n, n, beta, epsilon, D, c_n,
                              w0=0.5, w1=0.5, s_grid=None, m_grid_size=100):
    if s_grid is None:
        s_grid = np.arange(1, k_n + 1, max(1, k_n // 100))
    best_s, best_m, best_J = 1, 0, np.inf
    for s in s_grid:
        M = n - 2 * k_n + s
        m_grid = np.linspace(0, M, m_grid_size).astype(int)
        for m in m_grid:
            J = compute_J(s, m, k_n, n, beta, epsilon, D, c_n, w0, w1)
            if J < best_J:
                best_J = J
                best_s, best_m = int(s), int(m)
    return best_s, best_m, float(best_J)
''',

# ==================== src/methods/base.py ====================
"src/methods/base.py": '''"""方法基类"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AllocationResult:
    s_n: int
    m_n: int
    use_real_minority: bool = True


class BaseMethod(ABC):
    name: str = "base"

    @abstractmethod
    def allocate(self, k_n: int, n: int, **kwargs) -> AllocationResult:
        pass
''',

# ==================== src/methods/real_only.py ====================
"src/methods/real_only.py": '''from .base import BaseMethod, AllocationResult


class RealOnly(BaseMethod):
    name = "real_only"

    def allocate(self, k_n, n, **kwargs):
        return AllocationResult(s_n=0, m_n=0, use_real_minority=True)
''',

# ==================== src/methods/naive_balancing.py ====================
"src/methods/naive_balancing.py": '''from .base import BaseMethod, AllocationResult


class NaiveBalancing(BaseMethod):
    def __init__(self, tau: float):
        self.tau = tau
        self.name = f"naive_balancing_{tau}"

    def allocate(self, k_n, n, **kwargs):
        s = max(1, int(self.tau * k_n))
        m = n - 2 * k_n + s
        return AllocationResult(s_n=s, m_n=m, use_real_minority=True)


class NaiveBalancingOptimalS(BaseMethod):
    name = "naive_balancing_optimal_s"

    def __init__(self, beta, epsilon, D, c_n, w0=0.5, w1=0.5):
        self.beta = beta
        self.epsilon = epsilon
        self.D = D
        self.c_n = c_n
        self.w0 = w0
        self.w1 = w1

    def allocate(self, k_n, n, **kwargs):
        from ..allocation.optimization import find_theoretical_optimal
        s, _, _ = find_theoretical_optimal(
            k_n, n, self.beta, self.epsilon, self.D, self.c_n, self.w0, self.w1
        )
        m = n - 2 * k_n + s
        return AllocationResult(s_n=s, m_n=m, use_real_minority=True)
''',

# ==================== src/methods/theoretical_optimal.py ====================
"src/methods/theoretical_optimal.py": '''from .base import BaseMethod, AllocationResult
from ..allocation.optimization import find_theoretical_optimal


class TheoreticalOptimal(BaseMethod):
    name = "theoretical_optimal"

    def __init__(self, beta, epsilon, D, c_n, w0=0.5, w1=0.5):
        self.beta = beta
        self.epsilon = epsilon
        self.D = D
        self.c_n = c_n
        self.w0 = w0
        self.w1 = w1

    def allocate(self, k_n, n, **kwargs):
        s, m, _ = find_theoretical_optimal(
            k_n, n, self.beta, self.epsilon, self.D, self.c_n, self.w0, self.w1
        )
        return AllocationResult(s_n=s, m_n=m, use_real_minority=True)
''',

# ==================== src/methods/complete_replacement.py ====================
"src/methods/complete_replacement.py": '''from .base import BaseMethod, AllocationResult


class CompleteReplacement(BaseMethod):
    def __init__(self, s_ratio: float, name_suffix: str = ""):
        self.s_ratio = s_ratio
        self.name = f"complete_replacement_{name_suffix}" if name_suffix else f"complete_replacement_{s_ratio}"

    def allocate(self, k_n, n, **kwargs):
        s = max(1, int(self.s_ratio * k_n))
        m = n - 2 * k_n + s
        return AllocationResult(s_n=s, m_n=m, use_real_minority=False)
''',

# ==================== src/methods/fixed_s_optimal_m.py ====================
"src/methods/fixed_s_optimal_m.py": '''from .base import BaseMethod, AllocationResult
from ..allocation.optimization import find_optimal_m_fixed_s


class FixedSOptimalM(BaseMethod):
    def __init__(self, s_ratio: float, beta, epsilon, D, c_n, w0=0.5, w1=0.5):
        self.s_ratio = s_ratio
        self.beta = beta
        self.epsilon = epsilon
        self.D = D
        self.c_n = c_n
        self.w0 = w0
        self.w1 = w1
        self.name = f"fixed_s_{s_ratio}_optimal_m"

    def allocate(self, k_n, n, **kwargs):
        s = max(1, int(self.s_ratio * k_n))
        m, _ = find_optimal_m_fixed_s(
            s, k_n, n, self.beta, self.epsilon, self.D, self.c_n, self.w0, self.w1
        )
        return AllocationResult(s_n=s, m_n=m, use_real_minority=True)
''',

# ==================== src/methods/registry.py ====================
"src/methods/registry.py": '''from .real_only import RealOnly
from .naive_balancing import NaiveBalancing, NaiveBalancingOptimalS
from .theoretical_optimal import TheoreticalOptimal
from .complete_replacement import CompleteReplacement
from .fixed_s_optimal_m import FixedSOptimalM


METHOD_REGISTRY = {
    "real_only": RealOnly,
    "naive_balancing": NaiveBalancing,
    "naive_balancing_optimal_s": NaiveBalancingOptimalS,
    "theoretical_optimal": TheoreticalOptimal,
    "complete_replacement": CompleteReplacement,
    "fixed_s_optimal_m": FixedSOptimalM,
}


def build_method(name: str, **kwargs):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}")
    return METHOD_REGISTRY[name](**kwargs)


def register_method(name: str, cls):
    METHOD_REGISTRY[name] = cls
''',

# ==================== src/classifiers/logistic.py ====================
"src/classifiers/logistic.py": '''"""逻辑回归分类器"""
from sklearn.linear_model import LogisticRegression


def train_classifier(X_train, y_train, C=1.0, max_iter=1000):
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf
''',

# ==================== src/metrics/balanced_risk.py ====================
"src/metrics/balanced_risk.py": '''"""平衡风险指标"""
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


def balanced_log_loss(clf, X, y):
    """平衡对数损失"""
    proba = clf.predict_proba(X)
    mask0 = (y == 0)
    mask1 = (y == 1)
    loss0 = log_loss(y[mask0], proba[mask0], labels=[0, 1])
    loss1 = log_loss(y[mask1], proba[mask1], labels=[0, 1])
    return 0.5 * loss0 + 0.5 * loss1


def balanced_excess_risk(clf, X_test, y_test):
    """平衡超额风险 = R_bal(clf) - R_bal(oracle)"""
    oracle = LogisticRegression(max_iter=1000)
    oracle.fit(X_test, y_test)
    R_clf = balanced_log_loss(clf, X_test, y_test)
    R_oracle = balanced_log_loss(oracle, X_test, y_test)
    return R_clf - R_oracle
''',

# ==================== src/metrics/recall.py ====================
"src/metrics/recall.py": '''"""召回率和 F1"""
from sklearn.metrics import recall_score, f1_score


def minority_recall(clf, X, y):
    y_pred = clf.predict(X)
    return recall_score(y, y_pred, pos_label=1, zero_division=0)


def majority_recall(clf, X, y):
    y_pred = clf.predict(X)
    return recall_score(y, y_pred, pos_label=0, zero_division=0)


def macro_f1(clf, X, y):
    y_pred = clf.predict(X)
    return f1_score(y, y_pred, average="macro", zero_division=0)
''',

# ==================== src/experiment/runner.py ====================
"src/experiment/runner.py": '''"""单次实验运行器"""
import numpy as np
from ..synthesizer.registry import build_synthesizer
from ..classifiers.logistic import train_classifier
from ..metrics.balanced_risk import balanced_excess_risk, balanced_log_loss
from ..metrics.recall import minority_recall, majority_recall, macro_f1


def run_single_method(method, data_splits, synthesizer_cfg, classifier_cfg,
                      k_n, n, seed):
    """对单个方法、单次重复运行实验"""
    alloc = method.allocate(k_n=k_n, n=n)
    s_n, m_n = alloc.s_n, alloc.m_n

    X_min_pool = data_splits.X_min_train
    rng = np.random.default_rng(seed + 2000)

    if s_n > 0:
        idx = rng.permutation(len(X_min_pool))
        gen_idx = idx[:s_n]
        clf_idx = idx[s_n:]
        X_gen = X_min_pool[gen_idx]
        X_clf_real = X_min_pool[clf_idx] if alloc.use_real_minority else np.empty((0, X_min_pool.shape[1]))
    else:
        X_gen = None
        X_clf_real = X_min_pool if alloc.use_real_minority else np.empty((0, X_min_pool.shape[1]))

    # 生成合成样本
    if s_n > 0 and m_n > 0:
        synth = build_synthesizer(
            synthesizer_cfg["type"],
            mu_true=data_splits.mu_true if data_splits.mu_true is not None else np.zeros(X_min_pool.shape[1]),
            beta=synthesizer_cfg["beta"],
            epsilon=synthesizer_cfg["epsilon"],
            D=synthesizer_cfg["D"],
        )
        synth.fit(X_gen, seed=seed + 3000)
        X_syn = synth.sample(m_n, seed=seed + 4000)
        discrepancy = synth.discrepancy
    else:
        X_syn = np.empty((0, X_min_pool.shape[1]))
        discrepancy = np.nan

    # 组装训练集
    X_parts = [data_splits.X_maj_train]
    y_parts = [np.zeros(len(data_splits.X_maj_train))]
    if len(X_clf_real) > 0:
        X_parts.append(X_clf_real)
        y_parts.append(np.ones(len(X_clf_real)))
    if len(X_syn) > 0:
        X_parts.append(X_syn)
        y_parts.append(np.ones(len(X_syn)))

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    if len(np.unique(y_train)) < 2:
        return None

    clf = train_classifier(X_train, y_train, **classifier_cfg)

    return {
        "method": method.name,
        "s_n": int(s_n),
        "m_n": int(m_n),
        "s_ratio": s_n / k_n,
        "m_ratio": m_n / max(1, n - 2 * k_n),
        "discrepancy": float(discrepancy),
        "balanced_excess_risk": float(balanced_excess_risk(clf, data_splits.X_test, data_splits.y_test)),
        "balanced_log_loss": float(balanced_log_loss(clf, data_splits.X_test, data_splits.y_test)),
        "minority_recall": float(minority_recall(clf, data_splits.X_test, data_splits.y_test)),
        "majority_recall": float(majority_recall(clf, data_splits.X_test, data_splits.y_test)),
        "macro_f1": float(macro_f1(clf, data_splits.X_test, data_splits.y_test)),
    }
''',

# ==================== scripts/run_simulation.py ====================
"scripts/run_simulation.py": '''"""主实验脚本"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.data.generator import DataConfig, generate_gaussian_data
from src.data.split import SplitConfig, nested_split
from src.methods.registry import build_method
from src.experiment.runner import run_single_method
from src.utils.seed import set_seed


def build_methods(beta, epsilon, D, c_n):
    methods = []
    methods.append(build_method("real_only"))
    methods.append(build_method("naive_balancing", tau=0.1))
    methods.append(build_method("naive_balancing", tau=0.05))
    methods.append(build_method("naive_balancing", tau=0.5))
    methods.append(build_method("naive_balancing_optimal_s",
                                 beta=beta, epsilon=epsilon, D=D, c_n=c_n))
    methods.append(build_method("theoretical_optimal",
                                 beta=beta, epsilon=epsilon, D=D, c_n=c_n))
    methods.append(build_method("complete_replacement", s_ratio=0.5, name_suffix="ctrl"))
    methods.append(build_method("complete_replacement", s_ratio=1.0, name_suffix="opt"))
    methods.append(build_method("fixed_s_optimal_m",
                                 s_ratio=0.05, beta=beta, epsilon=epsilon, D=D, c_n=c_n))
    methods.append(build_method("fixed_s_optimal_m",
                                 s_ratio=0.1, beta=beta, epsilon=epsilon, D=D, c_n=c_n))
    methods.append(build_method("fixed_s_optimal_m",
                                 s_ratio=0.5, beta=beta, epsilon=epsilon, D=D, c_n=c_n))
    return methods


def main():
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("configs/experiment_grid.yaml", encoding="utf-8") as f:
        grid = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg["data"])
    split_cfg = SplitConfig(**cfg["experiment"])

    n = data_cfg.n
    k_n = data_cfg.k_n
    c_n = cfg["allocation"]["c_n"]
    D = cfg["synthesizer"]["D"]

    all_records = []

    for beta in grid["beta_grid"]:
        for epsilon in grid["epsilon_grid"]:
            print(f"\\n=== beta={beta}, eps={epsilon} ===")
            methods = build_methods(beta, epsilon, D, c_n)

            for rep in tqdm(range(cfg["experiment"]["n_repeats"])):
                seed = cfg["experiment"]["seed_base"] + rep
                set_seed(seed)

                X, y, mu = generate_gaussian_data(data_cfg, seed)
                splits = nested_split(X, y, split_cfg, seed)
                splits.mu_true = mu

                synth_cfg = {
                    "type": "gaussian",
                    "beta": beta,
                    "epsilon": epsilon,
                    "D": D,
                }
                clf_cfg = {"C": cfg["classifier"]["C"], "max_iter": cfg["classifier"]["max_iter"]}

                for method in methods:
                    res = run_single_method(
                        method, splits, synth_cfg, clf_cfg, k_n, n, seed
                    )
                    if res is not None:
                        res["beta"] = beta
                        res["epsilon"] = epsilon
                        res["rep"] = rep
                        all_records.append(res)

    df = pd.DataFrame(all_records)
    out_path = Path(cfg["output"]["results_dir"]) / "simulation_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\\nSaved to {out_path}")


if __name__ == "__main__":
    main()
''',

# ==================== scripts/plot_results.py ====================
"scripts/plot_results.py": '''"""结果可视化"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_phase_transition(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, eps in zip(axes, sorted(df["epsilon"].unique())):
        sub = df[df["epsilon"] == eps]
        for method in sub["method"].unique():
            data = sub[sub["method"] == method].groupby("beta")["balanced_excess_risk"].mean()
            ax.plot(data.index, data.values, marker="o", label=method, markersize=4)
        ax.set_title(f"eps = {eps}")
        ax.set_xlabel("beta")
        ax.set_ylabel("Balanced Excess Risk")
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "phase_transition.png", dpi=150)
    plt.close()


def plot_recall(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, eps in zip(axes, sorted(df["epsilon"].unique())):
        sub = df[df["epsilon"] == eps]
        for method in sub["method"].unique():
            data = sub[sub["method"] == method].groupby("beta")["minority_recall"].mean()
            ax.plot(data.index, data.values, marker="o", label=method, markersize=4)
        ax.set_title(f"eps = {eps}")
        ax.set_xlabel("beta")
        ax.set_ylabel("Minority Recall")
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "minority_recall.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("results/raw/simulation_results.csv")
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plot_phase_transition(df, "results/figures")
    plot_recall(df, "results/figures")
    print("Figures saved to results/figures/")
''',

}


def main():
    print(f"Project root: {PROJECT_ROOT}")
    
    # 创建所有目录和 __init__.py
    dirs_with_init = [
        "src", "src/data", "src/synthesizer", "src/allocation",
        "src/methods", "src/classifiers", "src/metrics",
        "src/experiment", "src/utils", "tests",
    ]
    for d in dirs_with_init:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / d / "__init__.py").touch()
    
    # 创建空目录
    for d in ["results/raw", "results/figures"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    
    # 写入所有文件
    for filepath, content in FILES.items():
        full_path = PROJECT_ROOT / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Created: {filepath}")
    
    print("\n" + "=" * 60)
    print("Project setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Verify installation: python -c \"import numpy, pandas, sklearn, yaml\"")
    print("  2. Run experiment: python scripts/run_simulation.py")
    print("  3. Plot results: python scripts/plot_results.py")


if __name__ == "__main__":
    main()