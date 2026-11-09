# This file is independently implemented for QuantumBridge SDK.
"""Report writers for QuantumBridge Router demo artifacts."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def write_router_demo_artifacts(payload: dict[str, Any], output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [
        _write_json(out / "router_decision_log.json", payload["decision"]),
        _write_json(out / "benchmark_report.json", payload["benchmark"]),
        _write_markdown(out / "benchmark_report.md", render_benchmark_markdown(payload["benchmark"])),
        _write_markdown(out / "poc_report_template.md", render_poc_template(payload)),
        _write_markdown(out / "router_api.md", render_router_api_markdown()),
        _write_markdown(out / "adapter_spec.md", render_adapter_spec_markdown()),
        _write_html(out / "index.html", render_demo_html(payload)),
    ]
    return written


def render_benchmark_markdown(report: dict[str, Any]) -> str:
    improvement = dict(report.get("improvement", {}))
    router = dict(report.get("router_summary", {}))
    lines = [
        "# QuantumBridge Router Benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"- Jobs: {report.get('job_count')}",
        f"- Router avg wait: {router.get('avg_wait')} seconds",
        f"- Router avg predicted fidelity: {router.get('avg_pred_fidelity')}",
        f"- Wait reduction vs round-robin: {improvement.get('wait_reduction_percent')}%",
        f"- Success target met: {improvement.get('success_target_met')}",
        "",
        "## Baselines",
        "",
    ]
    for name, summary in dict(report.get("baseline_summaries", {})).items():
        lines.append(f"- {name}: avg_wait={summary.get('avg_wait')}, avg_pred_fidelity={summary.get('avg_pred_fidelity')}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Local dry-run demo only.",
            "- No cloud access, token reads, or hardware submission.",
            "- Real backend adapter metadata is placeholder-safe until explicit credentials are supplied outside the repo.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_poc_template(payload: dict[str, Any]) -> str:
    improvement = payload["benchmark"]["improvement"]
    return "\n".join(
        [
            "# QuantumBridge Router PoC Report Template",
            "",
            "## Customer Question",
            "",
            "Can a fidelity-aware router reduce estimated queue wait while preserving acceptable fidelity on superconducting QPU backends?",
            "",
            "## Demo Evidence",
            "",
            f"- Wait reduction vs round-robin: {improvement.get('wait_reduction_percent')}%",
            f"- Success target met: {improvement.get('success_target_met')}",
            f"- Selected demo backend: {payload['decision'].get('selected_backend')}",
            "",
            "## PoC Data To Replace",
            "",
            "- Authorized backend metadata snapshot",
            "- Authorized calibration snapshot",
            "- Queue snapshot at submission time",
            "- Bell/GHZ/QAOA-small/VQE-small result counts",
            "- Customer-approved benchmark method",
            "",
            "## Commercial Next Step",
            "",
            "Run a two-week paid PoC to compare QuantumBridge Router against round-robin, least-queue, and best-fidelity-only policies.",
        ]
    ) + "\n"


def render_router_api_markdown() -> str:
    return """# QuantumBridge Router API

## POST /route

Input:

```json
{
  "qasm3": "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;",
  "sla": {
    "min_fidelity": 0.82,
    "max_wait": 900,
    "max_cost": 5,
    "shots": 1024,
    "allow_mock": true
  }
}
```

Output:

```json
{
  "selected_backend": "xianghan_superconducting_beta",
  "fallback_backend": "open_mock_gamma",
  "pred_fidelity": 0.91,
  "estimated_wait": 102.0,
  "estimated_cost": 0.56,
  "decision_trace": []
}
```

Boundary: local MVP only. It does not read tokens or submit hardware jobs.
"""


def render_adapter_spec_markdown() -> str:
    return """# Real Backend Adapter Spec

A production adapter should implement:

- `list_backends()`: returns backend id, qubit count, basis gates, coupling map, queue snapshot.
- `get_calibration(backend_id)`: returns authorized T1/T2, gate error, readout error, and timestamp.
- `submit_job(bundle, backend_id, shots)`: submits only when explicit credentials are provided outside the repo.
- `get_result(job_id)`: retrieves counts and status.

Security requirements:

- Store credentials in `.env.local` or a deployment secret manager.
- Never commit tokens, private endpoint URLs, internal calibration dumps, or non-public API docs.
- Keep dry-run mode available for demos and investor diligence.
"""


def render_demo_html(payload: dict[str, Any]) -> str:
    data = escape(json.dumps(payload, indent=2, sort_keys=True))
    improvement = payload["benchmark"]["improvement"]
    decision = payload["decision"]
    quafu_evidence = payload.get("quafu_evidence", {})
    quafu_decision = payload.get("quafu_router_decision", {})
    queue_rows = "\n".join(
        _render_queue_row(row)
        for row in quafu_evidence.get("chip_queues", [])
    )
    benchmark_rows = "\n".join(
        f"<tr><td>{escape(str(row['policy']))}</td><td>{row['avg_wait']}</td><td>{row['avg_pred_fidelity']}</td><td>{row['total_cost']}</td></tr>"
        for row in payload["benchmark"]["dashboard"]["benchmark_comparison"]
    )
    trace_rows = "\n".join(
        f"<tr><td>{escape(str(row['backend_id']))}</td><td>{row['estimated_wait']}s</td><td>{row['pred_fidelity']}</td><td>{'推荐' if row['selected'] else '备选'}</td></tr>"
        for row in quafu_decision.get("decision_trace", [])
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QuantumBridge Router | 量子真机任务路由 Demo</title>
  <style>
    :root {{
      --ink: #18202c;
      --muted: #5d6878;
      --line: #dfe5ec;
      --paper: #ffffff;
      --wash: #f4f6f8;
      --blue: #2563eb;
      --green: #16855c;
      --amber: #a15c00;
      --red: #b42318;
      --navy: #13233a;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--wash); color: var(--ink); }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ font-size: 40px; line-height: 1.12; margin: 0; letter-spacing: 0; max-width: 820px; }}
    h2 {{ font-size: 24px; margin: 0 0 14px; letter-spacing: 0; }}
    h3 {{ font-size: 16px; margin: 0 0 8px; letter-spacing: 0; }}
    p {{ line-height: 1.68; }}
    .topbar {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 10px 0 22px; color: var(--muted); font-size: 14px; }}
    .brand {{ font-weight: 760; color: var(--ink); }}
    .hero {{ display: grid; grid-template-columns: minmax(0, 1.18fr) minmax(360px, .82fr); gap: 22px; align-items: stretch; padding: 26px 0 18px; }}
    .hero-copy {{ padding: 28px 0; }}
    .eyebrow {{ color: var(--blue); font-weight: 760; margin-bottom: 12px; }}
    .lede {{ color: var(--muted); max-width: 820px; font-size: 17px; margin: 16px 0 0; }}
    .hero-panel {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 18px; display: grid; gap: 12px; }}
    .status-line {{ display: flex; justify-content: space-between; gap: 14px; border-bottom: 1px solid var(--line); padding-bottom: 12px; }}
    .status-line:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .status-label {{ color: var(--muted); font-size: 13px; }}
    .status-value {{ font-weight: 760; text-align: right; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 34px; }}
    .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 16px; min-height: 126px; }}
    .metric {{ font-size: 30px; font-weight: 800; margin-top: 10px; color: var(--navy); overflow-wrap: anywhere; }}
    .label {{ color: var(--muted); font-size: 13px; }}
    .section {{ margin: 20px 0 34px; }}
    .two-col {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .three-col {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .plain-panel {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }}
    .plain-panel p {{ margin: 0; color: var(--muted); }}
    .value-list {{ display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; }}
    .value-list li {{ border-left: 3px solid var(--blue); padding-left: 12px; line-height: 1.55; }}
    .flow {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; align-items: stretch; }}
    .step {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 118px; }}
    .step-number {{ width: 28px; height: 28px; border-radius: 50%; background: #e9f0ff; color: var(--blue); display: flex; align-items: center; justify-content: center; font-weight: 760; margin-bottom: 10px; }}
    .bar {{ display: flex; align-items: center; gap: 10px; min-width: 180px; }}
    .bar-track {{ height: 10px; background: #edf1f5; border-radius: 999px; flex: 1; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--blue); border-radius: 999px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--paper); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #edf0f4; font-size: 14px; vertical-align: middle; }}
    th {{ background: #eef3f8; color: #344054; font-weight: 760; }}
    tr:last-child td {{ border-bottom: 0; }}
    .tag {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; background: #eef3f8; font-size: 12px; color: #344054; font-weight: 700; }}
    .tag.green {{ background: #e7f7ef; color: var(--green); }}
    .tag.amber {{ background: #fff4df; color: var(--amber); }}
    .tag.red {{ background: #feeceb; color: var(--red); }}
    .callout {{ border-left: 4px solid var(--green); background: #eefaf4; padding: 16px; border-radius: 8px; }}
    .callout strong {{ color: var(--green); }}
    pre {{ overflow: auto; background: #111827; color: #e8edf7; padding: 16px; border-radius: 8px; font-size: 12px; line-height: 1.45; max-height: 360px; }}
    @media (max-width: 920px) {{
      .hero, .two-col, .three-col {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: 1fr 1fr; }}
      .flow {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 31px; }}
    }}
    @media (max-width: 560px) {{
      .grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 18px 14px 40px; }}
    }}
  </style>
</head>
<body>
<main>
  <div class="topbar">
    <div class="brand">QuantumBridge Router</div>
    <div>真机任务提交前的智能路由层</div>
  </div>

  <section class="hero">
    <div class="hero-copy">
      <div class="eyebrow">一句话价值</div>
      <h1>让量子真机用户在提交任务前，知道该选哪台芯片、为什么选、能省多少等待。</h1>
      <p class="lede">这不是另一个量子 SDK，也不是漂亮外壳。它是夹在用户和真机平台之间的决策层：读取授权任务队列、校准和历史结果，给出可审计的 backend 选择。</p>
    </div>
    <div class="hero-panel">
      <div class="status-line"><div><div class="status-label">当前 Demo 推荐</div><div class="status-value">{escape(str(quafu_decision.get('selected_backend', 'quafu_dongling')))}</div></div><span class="tag green">可解释</span></div>
      <div class="status-line"><div><div class="status-label">夸父任务页已验证字段</div><div class="status-value">chipName / status / submitTime</div></div><span class="tag">只读</span></div>
      <div class="status-line"><div><div class="status-label">样例队列不均衡</div><div class="status-value">{quafu_evidence.get('queue_imbalance_ratio', 1)}x</div></div><span class="tag amber">待换真实数据</span></div>
      <div class="status-line"><div><div class="status-label">Demo 基准等待时间下降</div><div class="status-value">{improvement.get('wait_reduction_percent')}%</div></div><span class="tag green">可量化</span></div>
    </div>
  </section>

  <section class="grid">
    <div class="card"><div class="label">给用户</div><div class="metric">少等</div><p>不用盲选芯片，先看队列和成功概率。</p></div>
    <div class="card"><div class="label">给平台</div><div class="metric">分流</div><p>减少热门芯片拥堵，提高整体任务吞吐。</p></div>
    <div class="card"><div class="label">给投资人</div><div class="metric">API</div><p>可做成 SaaS / 私有化 / 平台插件。</p></div>
    <div class="card"><div class="label">半年目标</div><div class="metric">PoC</div><p>用真实任务数据跑出节省等待的报告。</p></div>
  </section>

  <section class="section two-col">
    <div class="plain-panel">
      <h2>别人为什么会付钱</h2>
      <ul class="value-list">
        <li>量子真机资源稀缺，排队时间和失败任务都是成本；用户愿意为“更快拿到可用结果”付费。</li>
        <li>企业或高校实验室不想研究每台芯片状态，只想提交任务后得到稳定结果。</li>
        <li>真机平台需要提高利用率，不希望所有任务挤到同一台热门芯片。</li>
        <li>投资人能看懂的不是“量子很热”，而是“已有平台缺少提交前调度层”。</li>
      </ul>
    </div>
    <div class="plain-panel">
      <h2>这个产品到底卖什么</h2>
      <ul class="value-list">
        <li>Router API：输入 QASM / 任务要求，输出推荐 backend、备选 backend、原因和预计等待。</li>
        <li>PoC 报告：对比手动选择、轮询、最短队列、保真优先等策略。</li>
        <li>平台插件：嵌入 Quafu 类任务页，在提交前提示“推荐芯片”。</li>
        <li>私有部署：给研究院、企业量子团队接入内部真机或合作真机。</li>
      </ul>
    </div>
  </section>

  <section class="section">
    <h2>从用户提交任务到 Router 推荐</h2>
    <div class="flow">
      <div class="step"><div class="step-number">1</div><h3>用户提交线路</h3><p>QASM、shots、最低保真、最长等待时间。</p></div>
      <div class="step"><div class="step-number">2</div><h3>读取授权数据</h3><p>任务队列、芯片状态、校准数据、历史成功率。</p></div>
      <div class="step"><div class="step-number">3</div><h3>估算每台芯片</h3><p>等待时间、失败风险、成本和是否满足 SLA。</p></div>
      <div class="step"><div class="step-number">4</div><h3>选择 backend</h3><p>不是最快或最高保真单指标，而是综合最优。</p></div>
      <div class="step"><div class="step-number">5</div><h3>输出审计报告</h3><p>给用户、平台、投资人看的证据链。</p></div>
    </div>
  </section>

  <section class="section">
    <h2>夸父任务页证据链</h2>
    <div class="callout"><strong>已经确认：</strong>夸父任务页公开前端调用 <code>/api/tasks</code>，字段包含 <code>chipName</code>、<code>status</code>、<code>submitTime</code>。未登录访问返回 401，所以真实强证明需要授权登录后读取任务列表，不能伪造。</div>
    <table style="margin-top: 14px;"><thead><tr><th>芯片</th><th>活跃任务</th><th>状态分布</th><th>最老活跃提交时间</th><th>队列压力</th></tr></thead><tbody>{queue_rows}</tbody></table>
  </section>

  <section class="section two-col">
    <div>
      <h2>用任务列表做出的路由建议</h2>
      <table><thead><tr><th>候选 backend</th><th>预计等待</th><th>预测保真</th><th>结论</th></tr></thead><tbody>{trace_rows}</tbody></table>
    </div>
    <div>
      <h2>100 个模拟任务基准测试</h2>
      <table><thead><tr><th>策略</th><th>平均等待</th><th>平均预测保真</th><th>总成本</th></tr></thead><tbody>{benchmark_rows}</tbody></table>
    </div>
  </section>

  <section class="section three-col">
    <div class="plain-panel"><h3>第一笔钱怎么赚</h3><p>两周付费 PoC：接入一个平台或实验室任务数据，交付“队列节省 + 失败率分析 + 推荐策略”报告。</p></div>
    <div class="plain-panel"><h3>为什么大公司没做好</h3><p>大公司更偏平台和硬件入口，通常不愿承认“跨平台/跨芯片调度层”应该独立存在。</p></div>
    <div class="plain-panel"><h3>一人公司怎么开始</h3><p>先做只读数据接入和报告，不碰提交/取消任务；有证据后再做 API 和平台插件。</p></div>
  </section>

  <section class="section">
    <h2>原始审计数据</h2>
    <p class="lede">投资人或技术客户要追问时，可以展开这段 JSON 看每一步决策。正式 PoC 会把样例数据替换成授权真实数据。</p>
  <pre>{data}</pre>
  </section>
</main>
</body>
</html>
"""


def _render_queue_row(row: dict[str, Any]) -> str:
    active = int(row.get("active_tasks", 0))
    total = max(int(row.get("total_tasks", 0)), 1)
    width = min(100, int((active / max(total, active, 1)) * 100))
    status = ", ".join(f"{escape(str(key))}: {value}" for key, value in dict(row.get("status_counts", {})).items())
    pressure = "高" if active >= 3 else "中" if active > 0 else "低"
    tag = "red" if active >= 3 else "amber" if active > 0 else "green"
    return (
        "<tr>"
        f"<td>{escape(str(row.get('chip_name', 'UNKNOWN')))}</td>"
        f"<td>{active}</td>"
        f"<td>{status}</td>"
        f"<td>{escape(str(row.get('oldest_active_submit_time') or '-'))}</td>"
        f"<td><div class=\"bar\"><div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:{width}%\"></div></div><span class=\"tag {tag}\">{pressure}</span></div></td>"
        "</tr>"
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_markdown(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_html(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path
