from .real_only import RealOnly
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

# 哪些方法需要从当前 (beta, epsilon) 上下文注入 beta/epsilon/D/c_n
# 未列出的方法（如 real_only、naive_balancing、complete_replacement）
# 不接受这些参数，因此不注入，避免 TypeError。
NEEDS_CONTEXT = {
    "theoretical_optimal": ("beta", "epsilon", "D", "c_n"),
    "naive_balancing_optimal_s": ("beta", "epsilon", "D", "c_n"),
    "fixed_s_optimal_m": ("beta", "epsilon", "D", "c_n"),
}


def build_method(name: str, **kwargs):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}")
    return METHOD_REGISTRY[name](**kwargs)


def build_methods_from_grid(method_specs, context):
    """
    将 experiment_grid.yaml 中的 methods 列表展开为方法实例列表。

    method_specs: list of dict，每项形如
        - name: theoretical_optimal          # 可选，覆盖默认 name
          type: theoretical_optimal          # 必填，METHOD_REGISTRY 的键
          params: {s_ratio: 0.05, ...}       # 可选，传给类的固定参数
    额外规则：
      - 若 type 在 NEEDS_CONTEXT 中，自动从 context 注入 beta/epsilon/D/c_n。
      - 若指定了 name，覆盖 method.name，确保结果 CSV 中的列唯一。
      - 若 name 重复，直接报错，避免 pivot_table 把两个方法合并。
    """
    methods = []
    seen_names = set()

    for spec in method_specs:
        if "type" not in spec:
            raise ValueError(f"method spec 缺少 type: {spec}")
        type_ = spec["type"]
        if type_ not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method type: {type_}")

        params = dict(spec.get("params", {}))

        # 注入上下文参数
        for key in NEEDS_CONTEXT.get(type_, ()):
            if key in context:
                params[key] = context[key]

        method = build_method(type_, **params)

        # 覆盖 name
        if "name" in spec:
            method.name = spec["name"]

        if method.name in seen_names:
            raise ValueError(f"方法名重复: {method.name}，请检查 methods 列表")
        seen_names.add(method.name)

        methods.append(method)

    return methods


def register_method(name: str, cls):
    METHOD_REGISTRY[name] = cls