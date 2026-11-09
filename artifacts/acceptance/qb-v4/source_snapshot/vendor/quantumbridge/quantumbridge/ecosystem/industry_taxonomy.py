# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, Cirq, or third-party projects was copied.
"""Canonical industry capability taxonomy for QuantumBridge.

The four product categories are the stable user-facing architecture. Runtime,
compiler, algorithm, data, validation, and cloud services are shared foundations
that support these categories rather than additional product categories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


TAXONOMY_VERSION = "1.0"

SHARED_FOUNDATION = (
    "unified_ir",
    "circuit_and_qml_model",
    "compiler_and_transpiler",
    "algorithms_and_optimizers",
    "simulators_and_noise",
    "providers_jobs_and_results",
    "quafu_cloud_execution",
    "data_validation_and_provenance",
)


@dataclass(frozen=True)
class IndustryCapability:
    capability_id: str
    category_id: str
    category: str
    domain: str
    quantumbridge_scope: str
    qiskit_comparison: str
    cirq_comparison: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class IndustryCategory:
    category_id: str
    display_name: str
    capabilities: tuple[IndustryCapability, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "category_id": self.category_id,
            "display_name": self.display_name,
            "capabilities": [item.to_dict() for item in self.capabilities],
        }


def _capability(
    capability_id: str,
    category_id: str,
    category: str,
    domain: str,
    scope: str,
    qiskit: str,
    cirq: str,
) -> IndustryCapability:
    return IndustryCapability(capability_id, category_id, category, domain, scope, qiskit, cirq)


INDUSTRY_CATEGORIES = (
    IndustryCategory(
        "scientific_computing",
        "科学计算",
        (
            _capability(
                "quantum_chemistry",
                "scientific_computing",
                "科学计算",
                "量子化学",
                "分子哈密顿量、Hartree-Fock、UCCSD/VQE、基态与激发态、偶极矩、分子结构优化的统一工作流",
                "Qiskit Nature 提供电子结构、分子性质、活性空间、算子映射及求解接口",
                "OpenFermion 提供分子数据、费米子算子、映射与电路组件，并可结合 Cirq 执行",
            ),
            _capability(
                "quantum_biology_biomedicine",
                "scientific_computing",
                "科学计算",
                "量子生物／生物医药",
                "药物小分子能量与构象、蛋白局部活性空间、分子相互作用计算，并连接经典筛选与实验验证",
                "可复用 Qiskit Nature 化学能力；完整药物研发平台仍需领域工作流",
                "可复用 OpenFermion 化学组件；蛋白分区、药物筛选与业务流程需另行构建",
            ),
            _capability(
                "quantum_materials",
                "scientific_computing",
                "科学计算",
                "量子材料",
                "晶格与周期模型、能谱及能带相关计算，面向电池、催化剂、半导体和表面吸附",
                "Qiskit Nature 包含 Fermi-Hubbard、Heisenberg、Ising 等格点模型",
                "OpenFermion 支持材料相关费米子模型，并可结合 Cirq 模拟",
            ),
            _capability(
                "fundamental_physics_many_body",
                "scientific_computing",
                "科学计算",
                "基础物理与多体系统",
                "自旋模型、多体动力学、关联函数、简化核物理模型，并与化学、材料共享求解能力",
                "Qiskit Nature 的自旋、格点和多体建模可与算法库组合",
                "Cirq/ReCirq 提供 Fermi-Hubbard 和量子多体实验工作流",
            ),
        ),
    ),
    IndustryCategory(
        "finance",
        "金融",
        (
            _capability(
                "quantitative_investment_asset_allocation",
                "finance",
                "金融",
                "量化投资与资产配置",
                "组合选择、权重分配、预算与持仓约束、风险收益平衡、再平衡及结果验证",
                "Qiskit Finance 提供组合优化、组合分散化等模型与示例",
                "无对应专用金融包；通常基于通用电路和优化算法构建",
            ),
            _capability(
                "risk_measurement",
                "finance",
                "金融",
                "风险计量",
                "VaR/CVaR、信用风险、市场与流动性风险、尾部风险及压力测试",
                "Qiskit Finance 含信用风险、预期损失和 VaR/CVaR 示例",
                "可实现概率编码和相关量子算法；风险模型与业务验证需自行构建",
            ),
            _capability(
                "derivatives_pricing_hedging",
                "finance",
                "金融",
                "衍生品定价与对冲",
                "期权、固定收益及多资产估值，扩展路径模型、期货对冲和误差分析",
                "Qiskit Finance 含欧式期权、篮子期权、价差及固定收益定价示例",
                "具备算法实现基础，但没有成套金融定价模块",
            ),
            _capability(
                "financial_constrained_optimization",
                "finance",
                "金融",
                "金融约束优化",
                "将投资与交易约束转成 QUBO/Ising，并完成求解、解码和可行性检查",
                "Qiskit Finance 可与 Qiskit Optimization 组合使用",
                "可将问题编码成 Ising/QAOA；建模、约束处理及解码需自行组织",
            ),
            _capability(
                "correlation_market_volatility",
                "finance",
                "金融",
                "相关性与市场波动",
                "协方差、资产相关性、波动情景与混合 AI 模型，并连接数据与回测流程",
                "Qiskit Finance 提供市场数据处理和概率分布组件",
                "通用电路可用于研究；金融数据与波动模型需外接",
            ),
        ),
    ),
    IndustryCategory(
        "industrial_optimization",
        "工业优化",
        (
            _capability(
                "routing_production_scheduling",
                "industrial_optimization",
                "工业优化",
                "路径与生产调度优化",
                "TSP、车辆路径、时间窗、仓储物流、设备排程和作业车间调度",
                "Qiskit Optimization 提供 TSP、VehicleRouting 等应用模型",
                "具备 QAOA 等示例；路径约束、排程模型和业务接口需自行构建",
            ),
            _capability(
                "supply_chain_resource_allocation",
                "industrial_optimization",
                "工业优化",
                "供应链与资源配置",
                "供应网络、库存、产能、人力和设备配置，以及多周期供需协同",
                "可利用二次规划、背包、装箱等组件建模",
                "可执行编码后的优化电路；供应链模型与资源约束需自行构建",
            ),
            _capability(
                "energy_power_optimization",
                "industrial_optimization",
                "工业优化",
                "能源与电力优化",
                "电网负荷、机组组合、储能、新能源调度、能耗优化与物理可行性检查",
                "通用优化框架可承接适配模型；电网约束与系统接口需另行开发",
                "可用于能源优化算法研究；电力模型、约束及调度系统需外接",
            ),
            _capability(
                "general_combinatorial_optimization",
                "industrial_optimization",
                "工业优化",
                "通用组合优化",
                "QUBO/Ising、最大割、图优化、编码与解恢复，统一接入经典和量子求解器",
                "具有 QuadraticProgram、QUBO 转换器、MaxCut、QAOA 等组件",
                "具有 Ising、MaxCut、QAOA 示例；行业建模层需补充",
            ),
            _capability(
                "integer_multiobjective_decision",
                "industrial_optimization",
                "工业优化",
                "整数规划与多目标决策",
                "整数／混合整数、约束及多目标模型，并配套分解策略和经典基线",
                "支持二次规划建模、转换和经典求解器接口",
                "需外接数学建模与经典优化工具，再将适合部分交给量子电路",
            ),
        ),
    ),
    IndustryCategory(
        "quantum_security",
        "量子安全",
        (
            _capability(
                "post_quantum_cryptography_migration",
                "quantum_security",
                "量子安全",
                "后量子密码与迁移",
                "PQC 密钥封装、签名、参数评估与系统适配；UOV、SNOVA 等作为研究插件",
                "不属于 Qiskit 核心行业包；IBM Quantum Safe 是独立产品体系",
                "Cirq 是量子电路框架；PQC 实现和迁移工具需使用独立密码库",
            ),
            _capability(
                "quantum_secure_communication_qkd",
                "quantum_security",
                "量子安全",
                "量子保密通信／QKD",
                "BB84、噪声与窃听模型、误码率、密钥协商、纠错与隐私放大、链路仿真",
                "有 QKD 教学和电路实验；实际网络、设备与密钥管理需另行集成",
                "可基于电路与噪声仿真实现协议实验；不是现成 QKD 网络平台",
            ),
            _capability(
                "quantum_threat_cryptographic_assessment",
                "quantum_security",
                "量子安全",
                "量子威胁与密码安全评估",
                "Shor/Grover 小规模实验、经典密码量子资源估算、威胁模型及评测报告",
                "具有相关算法教学与实现基础；行业评估流程需扩展",
                "具有 Shor 等算法示例；完整安全评估体系需自行建设",
            ),
            _capability(
                "quantum_random_number_qrng",
                "quantum_security",
                "量子安全",
                "量子随机数／QRNG",
                "真实硬件熵源接入、健康检测、随机性提取和随机数服务",
                "可进行测量采样；安全级 QRNG 仍需物理熵源、检测和认证体系",
                "可进行测量与模拟采样；模拟随机数不等同于真实量子熵源",
            ),
            _capability(
                "data_security_privacy_computing",
                "quantum_security",
                "量子安全",
                "数据安全与隐私计算",
                "隐私计算协议研究、访问控制、数据完整性、运行审计与安全接口",
                "可构建相关协议电路；没有统一覆盖这些能力的 Qiskit 行业包",
                "可开展协议与电路实验；隐私计算系统和安全工程需要额外建设",
            ),
        ),
    ),
)


def industry_categories() -> tuple[IndustryCategory, ...]:
    return INDUSTRY_CATEGORIES


def industry_capabilities() -> tuple[IndustryCapability, ...]:
    return tuple(item for category in INDUSTRY_CATEGORIES for item in category.capabilities)


def industry_taxonomy() -> dict[str, object]:
    validate_industry_taxonomy()
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "product_categories": [category.to_dict() for category in INDUSTRY_CATEGORIES],
        "shared_foundation": list(SHARED_FOUNDATION),
        "comparison_ecosystems": ["Qiskit", "Cirq"],
    }


def validate_industry_taxonomy() -> bool:
    expected_categories = ("scientific_computing", "finance", "industrial_optimization", "quantum_security")
    category_ids = tuple(category.category_id for category in INDUSTRY_CATEGORIES)
    if category_ids != expected_categories:
        raise ValueError("QuantumBridge industry taxonomy must preserve the canonical four-category order")
    capabilities = industry_capabilities()
    capability_ids = [item.capability_id for item in capabilities]
    if len(capabilities) != 19 or len(capability_ids) != len(set(capability_ids)):
        raise ValueError("QuantumBridge industry taxonomy must contain 19 uniquely identified capabilities")
    for category in INDUSTRY_CATEGORIES:
        if not category.capabilities:
            raise ValueError("Every QuantumBridge industry category must contain capabilities")
        for capability in category.capabilities:
            if capability.category_id != category.category_id or capability.category != category.display_name:
                raise ValueError("Capability category metadata must match its parent category")
    return True
