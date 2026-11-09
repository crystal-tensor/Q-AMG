# QuantumBridge SDK

[Online user guide / 在线使用文档](https://crystal-tensor.github.io/QuantumBridge/readme.html?lang=zh)
covers installation, quickstart, API reference and tutorials for all 4 domains / 19 directions,
with searchable Chinese/English chapters and an interactive 3D Bloch-sphere explanation.

[Application framework / 应用框架](https://crystal-tensor.github.io/QuantumBridge/framework.html)
organizes four domains and 19 directions with executable research notebooks.
It is a scope and evidence browser, not a certificate of complete SDK parity.
For a local preview run `node studio/scripts/serve.mjs` and open
`http://127.0.0.1:8765/framework.html`.

### Real-Stock Finance Applications / 真实股票金融应用

[Open Finance / 打开金融分类](https://crystal-tensor.github.io/QuantumBridge/framework.html?category=finance&lang=zh)
· [中文使用文档](https://crystal-tensor.github.io/QuantumBridge/readme.html?lang=zh#finance)

| 金融方向 / Direction | 已执行案例 / Executed Case |
| --- | --- |
| 量化投资与资产配置 / Portfolio | [D-Wave CQM：真实股票组合](https://crystal-tensor.github.io/QuantumBridge/examples/finance/dwave_portfolio.html) |
| 金融约束优化 / Constraints | [D-Wave CQM：预算、持仓与行业约束](https://crystal-tensor.github.io/QuantumBridge/examples/finance/dwave_constraints.html) |
| 衍生品定价 / Pricing | [量子蒙特卡洛：IBM 股票校准的期权模型](https://crystal-tensor.github.io/QuantumBridge/examples/finance/qmc_pricing.html) |
| 风险计量 / Risk | [量子蒙特卡洛：历史损失与 VaR / CVaR](https://crystal-tensor.github.io/QuantumBridge/examples/finance/qmc_risk.html) |
| 相关性与市场波动 / Market Models | [可微混合模型：IBM 波动研究](https://crystal-tensor.github.io/QuantumBridge/examples/finance/qmc_hybrid.html) |

每页包含原理、实现步骤、代码、中间与最终输出，以及可下载的 Jupyter
Notebook。股票输入为真实 **2015 年历史数据**，不是实时行情；本批为本地
经典求解和量子仿真，**没有新增夸父或 D-Wave 云任务**，不代表真机验收。
[Cirq 金融方法与案例](https://crystal-tensor.github.io/QuantumBridge/examples/finance/cirq_finance.html)
另保留合成教学模型与 Google 官方通用量子方法之间的边界。

Install `pip install -e '.[finance-apps]'` in a separate environment. The finance
sections of the local framework now link two executed applications and their
downloadable notebooks: D-Wave CQM portfolio/budget constraints, and quantum
Monte Carlo pricing/risk with a separate differentiable hybrid-model example.

```python
from quantumbridge.finance import (
    build_portfolio_cqm, solve_portfolio, pricing_case, risk_case,
    pennylane_qmc, hybrid_volatility_case,
)

portfolio = solve_portfolio(build_portfolio_cqm(budget=220.))
pricing = pricing_case(shots=1024)
risk = risk_case(notional=1000., confidence=.95, shots=1024)
pl_price = pennylane_qmc(pricing["probabilities"], pricing["call_payoffs"])
pl_tail = pennylane_qmc(risk["probabilities"], risk["excess_values"])
pl_cvar = risk["raw_reference"]["VaR"] + pl_tail["estimate"] / (1-risk["confidence"])
hybrid = hybrid_volatility_case(epochs=15)
print(portfolio["optimum"], pricing["call"]["estimate"], risk["cvar_estimate"])
print("PennyLane QMC:", pl_price["estimate"], "CVaR:", pl_cvar)
```

These use verified real **2015 historical stock prices**, not live quotes or
synthetic stocks. The pinned Plotly snapshot, hash, price-basis caveats and
source-published MIT license are included in `quantumbridge/finance/data/`.
`dimod.ExactCQMSolver` is classical CPU enumeration; native QAOA and MLAE run on
QuantumBridge simulators. `pennylane_qmc` separately executes installed
PennyLane's matrix-based simulator template for pricing and historical tail-risk
expectations, with its phase-grid error shown separately. The hybrid model trains a native
QNode through PyTorch (Tape-adjoint on the published core, or simulator backprop
on queued-QNode development builds), records its actual backend and retains its out-of-sample baseline comparison even
when it does not improve it. Model option prices are not market option quotes.

No D-Wave cloud or Quafu hardware jobs are submitted. Native four-qubit
measurement-only QASM artifacts are available for review, but existing Quafu
layout/bit-order/accuracy acceptance gaps remain open. These bounded research
applications are not production finance, quantum advantage, or completion of
the full Qiskit/PennyLane core-fusion goal. Original synthetic cases and hardware
receipts are unchanged.

The 2026-10-27 development snapshot includes native public circuit/bit/register
contracts, Primitives V2, moving-power differentiation and MPS/TensorFlow work,
plus standard QPY13 control-flow and scoped-variable transport. This is still
work in progress, not complete Qiskit/PennyLane parity. The publication check
passed 1305 focused tests with 3 existing expected failures, and all 7 page
navigation/export checks. Desktop/mobile home-case-home navigation covered all
19 cases, with notebook downloads. A separate full-suite run stalled and was
stopped; no full-suite pass or current global coverage is claimed. Four known
legacy primitive-container contract conflicts remain unresolved (the separate
primitive suite returned 156 passed and 4 failed). Private
proposals, credentials, environments and temporary diagnostics are not published.

Stage 12-B P4 is in progress: native Gate definitions, structural decomposition,
repeat/tensor and symbolic controlled phase now have execution tests. Complete
Gate/QASM/QPY/annotation contracts and P4 final acceptance are still pending;
this does not change the incomplete full Qiskit/PennyLane replacement status.

P4-S development adds executable custom Gate IR records, real QASM gate
declarations and standard QPY v13 custom-definition binary records. Supported
paths retain native execution and derivatives. This is not complete P4 or
full-format parity: source cannot preserve UUID/object metadata, QASM2 rejects
nonzero internal phase, and unsupported QPY records remain explicit. Run the
SDK-free demonstration with
`PYTHONPATH=. python examples/native_custom_gate_interchange.py`.

P4-T development adds native ordered `AnnotatedOperation` modifiers and
resolution through the existing Gate/compiler/device/gradient engine. Native
IR keeps unresolved structure; `standard_gate_definition` and
`standard_gate_object` expose 31 phase-exact native standard recipes. These
builders are not public Qiskit standard-gate subclasses or their mutable ABI.
QASM rejects unresolved annotations; explicitly resolving to a Gate changes
the object structure. P4-U1 now implements standard QPY v13 `a`/`m` binary
records for supported unitary annotations, including nested/ordered modifiers,
open controls, custom/standard bases, symbolic base parameters and phase.
This is independent native code, not an upstream SDK serialization wrapper.
Symbolic modifier powers are unsupported: v13 stores the exponent as float64.
Public Qiskit 2.4.1 and 2.6.0 themselves lose the enclosing circuit parameter
index on v13 annotation reload; object parameters survive, but this is an
explicit unresolved public binding ABI gap, not claimed lossless interop.
The original U1 subset rejected standalone custom Instruction records and
nonunitary bases; later U3B/R2 support is described below. Arbitrary object
metadata and unsupported labels still reject. Dedicated standard return
classes and full original P4 acceptance remain unfinished.
SDK-free numeric demonstrations are
`examples/native_annotated_gate.py`, `examples/native_gate_modifier_gradient.py`
and `examples/native_standard_gate_decomposition.py`. P4-T/P4 and full SDK
replacement remain incomplete; tests are not a percentage of feature parity.

P4-U2 work adds a live checked `params` view for Gate/Instruction/annotations,
real `validate_parameter`, atomic definition rebinding and owned copy isolation.
It is a MutableSequence proxy, not a literal public Python list or a detached
list copy. Native formal binding records retain further mutation through native
IR, including numeric-bound copies. Generic Instruction payload validation is
permissive; physical Gate angles are real and finite. The standard QPY subset
still preserves current structure/math, not arbitrary private editable history
of an already-numeric custom Gate. Changed bound definitions without explicit
symbolic lineage reject further ambiguous rebinding instead of guessing from
equal constants. Dedicated public standard Gate classes and exact error ABI,
dynamic/empty contracts and original P4 joint acceptance remain unfinished.
The SDK-free executable example is `examples/native_mutable_gate_parameters.py`.
The exact U2 runtime passed three FULL runs:4263/5291/5289 tests in SDK-free,
main and alternate environments. Explicit global line coverage is
30511/33474=91.1484%, not a feature-parity percentage; exclusions remain68.
Independent JAX/TF checks are FOCUS, not additional FULL runs. See the
[checkpoint report](docs/implementation/stage12_b_p4u2_mutable_parameter_progress.md)
for exact input hashes, receipts and remaining contracts. No new push/remote
CI or complete original28-section matrix is claimed.

P4-U3 work adds native public standard Gate objects, not aliases to upstream
SDK classes: RX/RY/RZ, Phase/U, H/X/Y/Z, CX/CY/CZ/CH/Swap/CSwap,
CRX/CRY/CRZ/CPhase, CCX/CCZ/CU and owned UnitaryGate. The existing engine
executes their definitions, controlled phase, mutable bindings and derivatives.
Numeric generic powers now return a real native UnitaryGate; annotation
lowering keeps structural integer powers for MPS/resource safety. Native IR
uses explicit class records; QPY uses actual standard class fields and bounded
NumPy records, including bilateral matrix wire order. Known standard source
forms preserve coherent phase instead of becoming lossy custom declarations.
Public singleton identity, canonical definition layouts, all Instruction
method signatures and all remaining P4 contracts are not claimed identical.
The supported standard-gate checkpoint passes27 independent local gates:
SDK-free/main/alternate FULL4473/5728/5726 passed, global line coverage91.2022%,
exclusions68. These are test/coverage counts, not feature-completion percentages.
Instruction constructor/inverse/repeat and Circuit return ABI still need U3
work. This is IN_PROGRESS, not U3/P4/full SDK acceptance. Run the SDK-free example
`examples/native_standard_gate_contracts.py`; see the
[U3 checkpoint report](docs/implementation/stage12_b_p4u3_standard_gate_progress.md)
for true red/green evidence and final-gate status.

P4-U3B work now implements native Instruction label/inverse/repeat contracts,
owned repeated Instruction objects, explicit IR and real QPY custom Instruction
records. Circuit inplace compose/binding returns None after mutation; inverse
and repeat naming follow separately measured public rules. Native lowering,
four-device measurement/control execution, framework gradients and MPS reuse
the existing engines. Initial10 red cases recover. All29 frozen checkpoint
gates pass: SDK-free/main/alternate FULL4544/5900/5898 passed; GLOBAL coverage
31253/34235=91.28961589017088%, exclusions68. This validates a supported subset,
not complete U3 or full SDK parity. Zero-width repetition, classical QPY custom
Instruction headers, source reference declarations and other full public ABI
contracts remain incomplete. See the
[Instruction/Circuit progress report](docs/implementation/stage12_b_p4u3b_instruction_circuit_abi_progress.md)
and run `PYTHONPATH=. python examples/native_instruction_circuit_contracts.py`.

P4-U3C work adds native Circuit wrap/copy composition, checked live operation
owners, typed Var/Stretch scopes, identifier remapping and capture inlining
through the existing dynamic executor. Integer/negative/slice quantum and
classical maps and real parameter/phase/derivative execution are covered by
focused tests. All31 frozen checkpoint children exit0: SDK-free/main/alternate
FULL4646/6055/6053 passed; main GLOBAL31678/34681=91.34108013033072%,
exclusions68 unchanged. The original28-section local matrix also exits0 in
the SDK-free environment; unavailable optional subsets and pytest-cov were
not executed there. These results are not complete U3 or full SDK parity.
Duration execution and typed-scope
QPY/QASM transport remain unsupported, not metadata-only substitutes. Run the
SDK-free `PYTHONPATH=. python examples/native_circuit_composition.py` and see
the [composition progress report](docs/implementation/stage12_b_p4u3c_composition_classical_scope_progress.md).

P4-U3D is in development: native Bit/Register identity and aliases now feed
actual execution operands. Native QuantumCircuit provides checked public
append/builders, CircuitInstruction contexts and InstructionSet handles while
Circuit retains fluent returns, sharing the existing authoritative store and
engines. Standard invocation parameter frames, real QPYv13 normal/loose/alias/
external-register records, and representable QASM register-array declarations
have targeted numeric tests. Unsupported source aliases, ancilla QPY and typed
scope transport remain explicit. These are supported subsets, not complete
QuantumCircuit, QPY or SDK parity. No new frozen FULL/GLOBAL acceptance is
claimed. Run the SDK-free examples `examples/native_public_quantumcircuit.py`
and `examples/native_bit_register_qpy_gradient.py`. See the
[bit/register work log](docs/implementation/stage12_b_p4u3d_bits_registers_progress.md)
and the SDK-free `examples/native_bit_register_identity.py` example.

U3D2-R development extends cached invocation frames to native generic Gate and
Instruction owners, checked data replacement and indexed handle inversion.
Public copy/empty-copy/repeat/power/tensor/decompose outputs now retain actual
native stores and have independent phase/parameter/execution tests. Scalar
1x1 UnitaryGate matrices have real zero-qubit execution, coherent control and
bilateral standard QPY tests. The SDK-free numeric example is
`examples/native_public_circuit_frames.py`. Native generic inverse/compose do
not reproduce upstream stale owner/definition states that cannot execute;
this is an explicit observable difference, not full public ABI parity.
Canonical decomposition recipes, factory naming/metadata, all remaining
operation classes and typed public contexts still need work. Current tests
are development FOCUS, not new frozen FULL/GLOBAL/matrix acceptance.

U3D2-R2 adds a measured public binding subset: Parameter/string/vector mappings,
ordered iterables, flat_input/strict, inplace return rules, non-inplace names
and UUID-preserving get_parameter/has_parameter queries. Public naming is
kept out of internal definition binding, so native IR formal binding plans
remain deterministic and editable. Invalid binds remain atomic, with managed
callback rejection and bounded iterable/vector admission. These additions do
not complete public metadata or all conversion/decomposition contracts.
The later ParameterView continuation below adds the measured public snapshot
container. Full SDK parity and final frozen
acceptance are still pending.

Public Gate/Instruction conversion now creates actual local default q/c bit
owners and keeps complete formal maps, UUIDs, operand order and coherent phase.
Native QiskitError remains a ValueError-compatible extension, with no upstream
exception dependency. The custom-layer selective decompose subset is tested
for names, labels, wildcard/type/mixed selectors and one layer per repetition,
with definition/selector admission before copying. QPY can structurally export
the supported doubly controlled X/Z/Swap and same-axis rotations; the latter
use five native primitives without dense export. Public2.4.1's mapped
Instruction-phase bug is recorded, not copied; actual2.6.0 behavior matches.
Public metadata now has a separate user-data domain: setting a user key named
global_phase, classical_scope or registers cannot change executable state.
Constructor/setter reference, deep-copy factories and empty-metadata factories
are measured against actual public 2.4.1/2.6.0 calls. Decompose currently follows
2.4.1's shared dictionary; 2.6.0's shallow outer-copy difference is explicit.
Native IR retains typed execution state; standard QPY carries JSON user data
separately from real phase/register records. Non-JSON objects remain valid
in memory but explicitly refuse unsupported portable encoding. QPY-decoded
internal register descriptions are no longer exposed as user metadata; real
register owners remain available. Legacy fluent Circuit metadata configuration
is unchanged. All dynamic/annotation/control and canonical definition recipes
and the final frozen acceptance are incomplete.

The R2 owner-lifecycle continuation extends captured generic invocation frames
to ControlledGate and AnnotatedOperation: later source parameter mutation does
not replace captured executable slots. Binding, explicit data replacement,
inverse/copy/compose and native IR/QPY have full complex-matrix and derivative
tests, including 100-wire MPS without dense expansion and four AD interfaces.
Generic controlled names retain the actual open-control suffix, including
literal suffixes in user-assigned names. Real QPYv13 custom Instruction and
annotation records now carry supported classical widths and operand mappings.
Measurement/reset annotations can be constructed and transported but illegal
unitary resolution is refused. A measured custom Instruction roundtrip runs on
the four native local devices; this does not establish real cloud execution.
These are development subsets, not final frozen acceptance, complete QPY,
complete public ABI or full Qiskit/PennyLane functional parity.
Definition replacement and nested graph mutation now also have independent
complex-matrix/factory/transport tests. Conversion-derived Instruction names
update atomically, and public copy rejects malformed/cyclic/over-budget raw
definitions before copying. Actual public ControlledGate inverse can use a
stale base or fail after definition replacement; native mathematical consistency
is retained as an explicit intentional difference, not identical public ABI.

The controlled-base continuation tests two/open/nested control ownership against
two actual public versions and independent complex-matrix oracles. Supported
multi-control rotation/fixed-gate QPY export is structural, not a dense matrix
escape. Variable-width MCPhaseGate/MCXGate import uses native generic controlled
objects; dedicated public subclasses are not implemented by this change.
Native controlled derivatives are zero outside the active subspace and use the
correct half-frequency shifts for controlled Pauli rotations. Tests exercise
Jacobian/Hessian/HVP, four AD interfaces and 100-wire nested-control MPS execution.
Native IR now explicitly preserves invalidated numeric formal history and the
actual edited definition, while still refusing ambiguous subsequent rebinding.
Labeled open CCX public QPY reload and symbolic annotation parameter indexing
have measured upstream failures; they are not claimed lossless public transport.
Same-name/different-UUID native update failures remain atomic even where measured
public updates are not. These are supported development slices, not original R2
frozen acceptance or full SDK replacement. No cloud/hardware execution is proved.

Public QuantumCircuit.parameters now returns an independent native ParameterView
snapshot, with ordered iteration/indexing/list slices, mutable data/remove,
copy semantics, set comparisons and the measured reflected/augmented operators.
Intersection preserves order and duplicates; union, difference and symmetric
difference return sets. Snapshot mutation cannot change circuit parameters,
captured invocation slots, formal binding plans, IR/QPY or real derivatives.
Both installed public reference versions are called for these contracts; no
upstream implementation is used at runtime. Logical parameter/work bounds and
managed foreign-callback admission apply before copying or changing the view.
Legacy fluent Circuit.parameters remains a tuple. This closes a specific public
collection gap, not all Parameter/ParameterVector representations, public circuit
ABI, original frozen acceptance or full Qiskit/PennyLane functional parity.

The subsequent vector audit adds a read-only vector name, live list parameters,
list slices, stable-UUID shrink/regrowth, native integral indices and lengths,
empty vector names and atomic, bounded growth. Managed scalar/vector input
admission precedes foreign truth/index/equality callbacks. Both native IR and
standard QPY retain empty-name vector identities and actual bindings. Native
list mutation keeps length/iteration/params coherent, matching measured2.4.1;
measured2.6.0 can diverge after manual params-list mutation, an explicit safety
difference. Negative lengths are refused rather than copying2.4.1's negative
slice truncation. Public UUID objects/complete scalar-name/representation ABI
and general noninteger Operator powers remain distinct unfinished requirements.

Native symbolic mathematics passed bounded local acceptance in Stage 12-B P3. Scalar
functions, symbolic/repeated differentiation and simultaneous substitution now
feed the native Circuit/Instruction and quantum derivative paths; this is not
yet full ParameterExpression or full Qiskit/PennyLane replacement. Final
SDK-free/main/alternate FULL runs have 3875/4752/4750 passed; the complete
28-section local matrix exits 0. Line coverage is 90.7832%, not functional
completion. No push or new-SHA remote CI is claimed. See the
[gap audit](docs/implementation/stage12_b_p3_symbolic_math_gap_audit.md) for
return-contract, domain, serialization and validation boundaries. The new
`examples/native_symbolic_*.py` examples run against the checkout with
`PYTHONPATH=. python examples/native_symbolic_functions.py`.

Clean-room ecosystem parity for experimental quantum workflows.

QuantumBridge is an experimental quantum SDK. The current repository keeps the P1 release-candidate baseline intact while Stage 9 promotes selected ecosystem lanes from inventory/schema coverage to bounded executable slices.

QuantumBridge is an independent project. It is not an official Qiskit, PennyLane, IBM, or Xanadu project, and it does not claim full feature parity or full replacement coverage.

## Current Status

- Experimental SDK.
- P1 release-candidate subset implementation.
- Apache-2.0 compliance route with attribution records.
- Qiskit and PennyLane support is provided through optional adapter / compatibility layers.
- Qiskit and PennyLane remain separate projects owned by their respective rightsholders.
- Wider ecosystem packages are optional dependencies only; QuantumBridge does not vendor third-party package source.

## Supported P0 Features

- Native circuit construction with X, Y, Z, H, RX, RY, RZ, CX, CZ.
- Statevector simulation.
- Shot sampling.
- Result serialization.
- PauliString, Hamiltonian, and SparsePauliOperator basics.
- Statevector and DensityMatrix information helpers.
- Sampler and Estimator primitives.
- Parameter-shift and finite-difference gradients.
- Simple VQE and QAOA.
- OpenQASM-style subset import/export.
- Qiskit basic circuit import/export and counts adapter when Qiskit is installed.
- PennyLane observable/tape/executable subset bridge when PennyLane is installed.
- NumPy/Torch/JAX array interface helpers, with Torch and JAX optional.
- Minimal QNode-like wrapper and QML templates.

## Native QNode Differentiation

`quantumbridge.qml.NativeQNode` is the existing `QNode` with native first-order
`gradient`, `jacobian`, VJP/JVP and execute-and-derivative methods. Gate-occurrence
parameter-shift and pure-vector adjoint preserve shared-input/classical chain
rules. Torch backward and Autograd grad use QuantumBridge quantum derivatives,
not PennyLane execution. Shot vectors run separate seeded sampled partitions.

```python
from quantumbridge import PauliZ
from quantumbridge.qml import NativeQNode

def program(tape, theta):
    tape.ry(theta, 0)
    tape.ry(theta, 0)
    return tape.expval(PauliZ(0))

node = NativeQNode(program, 1, diff_method="adjoint")
value, gradient = node.execute_and_compute_derivatives(0.4)
```

See [native training](examples/native_qnode_gradient.py),
[Torch training](examples/native_qnode_torch_training.py),
[shot partitions](examples/native_qnode_shot_vectors.py) and the
[Stage 11K report](docs/implementation/stage11k_native_qnode_autodiff_report.md).
Full autodiff parity is not complete: simulator backprop, complete JAX/TF, higher order,
capture and unsupported dynamic/adjoint cases remain explicit gaps. MPS gradients
retain tensor execution and truncation/resource warnings. Legacy mixed analytic
state plus sampled queries emit a warning and record actual measurement shots.

## Native Dynamic Control Flow

Native circuits now execute typed Bool/Bit/UInt expressions, runtime storage,
structured if/else, for/while, switch and loop break/continue on the native
statevector, density, MPS and supported Clifford tableau paths. Native Tape
also provides mid-circuit measure/reset and conditional callbacks. Examples:

```bash
python -m examples.native_dynamic_control_flow
python -m examples.native_measurement_feedforward
python -m examples.native_mps_dynamic_100_qubits
```

ExecutionLimits bounds loops, executed instructions, control depth, branches
and shots. UInt arithmetic wraps modulo its width; symbolic conditions cannot
silently become Python constants. Structured IR preserves executable expression
trees. Dynamic routing/scheduling and complete PennyLane capture/autodiff remain
open; unsupported compilation is rejected explicitly. See the
[Stage 11J report](docs/implementation/stage11j_native_dynamic_control_flow_report.md)
and [core parity ledger](docs/compat/core_parity_ledger.md).

## Native Matrix Product State Simulation

`MatrixProductStateDevice` and `MatrixProductStateBackend` execute native
tensor contractions, nonadjacent gates, channels, reset and classical feedback.
They integrate with Backend/Job, V2 primitives and the native QNode. The example
runs a 100-qubit GHZ circuit, observables, subsystem reductions, sampling and
compressed JSON output without expanding a dense statevector:

```bash
python -m examples.native_matrix_product_state_quantumbridge
```

Bond limits/cutoff can approximate the state; discarded Schmidt weights are
reported and are not a final-error guarantee. Unmeasured analytic Result keeps
a compressed MPS/ensemble rather than an implicit full probability table.
Dense output and entanglement growth remain resource-limited. This is not
complete Aer/PennyLane plugin or save-instruction parity. See the
[Stage 11I report](docs/implementation/stage11i_native_matrix_product_state_report.md).

## Native Pauli Algebra

The native `Pauli` and `PauliList` implementations now retain the full
`1, -1, i, -i` phase, support multi-qubit operators and subsystem composition,
and execute as observables in native state and QNode workflows. Seeded
`random_pauli` and `random_pauli_list` are also available without installing
Qiskit or PennyLane.

Run the native example with:

```bash
python -m examples.native_pauli_algebra_quantumbridge
```

API migration: `Pauli("X") @ Pauli("Y")` now returns `Pauli("iZ")`.
The previous tuple result remains available through
`Pauli("X").multiply(Pauli("Y"))`. Pauli-list composition now broadcasts rows,
retains phase, and uses Qiskit's multiplication order. Use
`commutation_matrix()` for all pairwise relations and `commutes()` for
broadcasted rows. Symplectic arrays and Pauli subsystem indices put qubit zero
on the right; native Circuit/QML wire indices continue to put wire zero first.
The executable scope and remaining gaps are recorded in the
[Stage 11E report](docs/implementation/stage11e_native_pauli_algebra_report.md).

## Native Clifford And Stabilizer Execution

`Clifford` now uses a signed binary symplectic tableau, validates that gates
are Clifford, and supports composition, inverses, tensor products and native
circuit synthesis. `StabilizerState` executes Clifford evolution, Pauli
expectations, projective measurement, reset and sampling without allocating
a dense statevector. `random_clifford` samples the Clifford group using native
binary algebra. None of these require Qiskit or PennyLane at runtime.

```bash
python -m examples.native_clifford_stabilizer_quantumbridge
```

The example executes a 64-qubit GHZ state using a 16,512-byte tableau.
Tableau and stabilizer subsystem indices use rightmost-qubit-zero ordering;
native Circuit wires are translated from their existing wire-zero-first
ordering. Dense conversion is explicit and exponential. Clifford composition
now follows Qiskit's order: `a.compose(b)` represents `b * a`, while
`a.dot(b)` represents `a * b`. Non-Clifford and classically controlled circuits
are rejected. This is not full Aer stabilizer-backend parity or general noisy
simulation. See the [Stage 11F report](docs/implementation/stage11f_native_clifford_stabilizer_report.md).

## Ecosystem Compatibility Status

Stage 11H adds native `DensityMatrixDevice` and
`Provider().get_backend("density_matrix")` / `DensityMatrixBackend` execution.
Bound non-Clifford gates, in-circuit multi-qubit CPTP Kraus channels, reset,
classical feedback, mixed initial states, Backend/Job/V2 primitives and native
QNode mixed-state outputs are callable without Qiskit or PennyLane. Density
matrices are first-class serializable Result data, not mislabeled statevectors.

```bash
python -m examples.native_mixed_state_quantumbridge
```

PhaseDamping now uses the standard parameter convention: off-diagonal terms
are multiplied by `sqrt(1-p)`, not `1-p`. To reproduce the old attenuation at
`p_old`, use `p = 2*p_old - p_old*p_old`. NoisyEstimator now evaluates X/Y and
Hamiltonian observables on the noisy density matrix. Dense simulation remains
exponential and enforces a configurable memory bound. Scope and remaining
gaps are in the [Stage 11H report](docs/implementation/stage11h_native_mixed_state_execution_report.md).

Stage 11G adds an executable native `StabilizerDevice` and
`Provider().get_backend("stabilizer")` / `StabilizerBackend` path. Backend/Job,
Sampler V2 and Estimator V2 can run large Clifford circuits without enumerating
all basis probabilities. Ordered mid-circuit measurements, reset, nested
if/else feedback, classical memory and native Clifford-mixture/readout noise
are exercised, including optional Aer comparisons. Native result bitstrings
remain wire-zero-first or classical-bit-zero-first, not Qiskit's reversed
printed-register order. Run:

```bash
python -m examples.native_stabilizer_backend_quantumbridge
```

This adds a real stabilizer execution path, not complete Aer, dynamic-circuit,
noise-model or primitive parity. The exact scope, compiler boundaries and
migration notes are in the [Stage 11G report](docs/implementation/stage11g_native_stabilizer_backend_report.md).

Stage 7 planning defines function coverage by level:

- Level 0 Inventory: public API names are identified and written to a matrix; this does not mean runnable support.
- Level 1 Passthrough: when the upstream optional dependency is installed, QuantumBridge can retrieve or call selected upstream objects and wrap results with provenance.
- Level 2 Adapter: selected upstream objects convert into QuantumBridge schemas, such as Circuit, IR, Result, Hamiltonian, Problem, Job, or Backend.
- Level 3 Native subset: QuantumBridge implements an independent subset for core capabilities.
- Level 4 Production equivalent: not promised in this project stage.

Current Qiskit ecosystem status:

- Qiskit core: Level 0 inventory, Level 1 passthrough contract, and a reviewed Level 2 subset for basic circuits/results.
- Qiskit Aer: Level 0/1 inventory and passthrough, Level 2 result wrapper, and a Stage 9F Level 3 educational native simulator subset for small statevector, qasm-style counts, and simple sampling-noise examples. Missing optional packages report unsupported metadata.
- Qiskit Nature: Level 0 inventory, Level 1 passthrough contract, Level 2 result schema wrappers, and a Stage 9D Level 3 educational native executable H2 / LiH chemistry subset. Missing optional packages report unsupported metadata.
- Qiskit Algorithms: Level 0/1 inventory and passthrough, Level 2 result wrappers, and a Stage 9C Level 3 educational native executable subset for VQE, QAOA-compatible MaxCut, and Grover examples. Missing optional packages report unsupported metadata.
- Qiskit Finance: Level 0/1 inventory and passthrough, Level 2 result wrapper, and a Stage 9A Level 3 educational portfolio-optimization native subset for deterministic four-asset mean-variance examples. Missing optional packages report unsupported metadata.
- Qiskit Optimization: Level 0/1 inventory and passthrough, Level 2 result wrappers, and a Stage 9B Level 3 educational native `QuadraticProgram` subset for small binary optimization examples. Missing optional packages report unsupported metadata.
- Qiskit Machine Learning: Level 0 inventory and contract metadata; missing optional packages report unsupported metadata.
- Qiskit Dynamics, Experiments, Metal, Runtime, and Addons: advisory or offline-only contracts with explicit warnings.
- Qiskit Runtime remains offline-only: no IBM Cloud access, no token reads, no credential storage, and no job submission.
- Qiskit Metal remains advisory: no chip fabrication, external EM simulation, or layout signoff support is claimed.
- Mitiq / error mitigation: Stage 9G Level 3 educational native zero-noise extrapolation and readout mitigation, plus optional upstream Mitiq dependency metadata when installed. This is not production error mitigation and not a complete Mitiq replacement.
- PennyLane-Qiskit bridge: Stage 9H Level 3 educational native bidirectional bridge for a basic-gate subset, including Qiskit circuit to PennyLane executable spec, PennyLane operation/tape metadata to Qiskit circuit, and Bell-state equivalence proof. This is not a full PennyLane-Qiskit plugin replacement and not full Qiskit or PennyLane parity.
- MQT Core / DDSIM / QMAP compatibility: Stage 9J Level 3 educational native circuit/IR compatibility, DDSIM-like small-circuit simulation, decision-diagram-inspired metadata, and QMAP-like topology routing with SWAP insertion. This is not a full MQT replacement, not decision-diagram parity, and not production compiler/simulator/mapper parity.
- TorchQuantum / PyTorch-style QML compatibility: Stage 9K Level 3 educational native quantum layer, tensor/batch forward, deterministic toy classifier training, optional torch tensor path, and optional upstream TorchQuantum boundary. This is not a full TorchQuantum or PyTorch replacement, not production QML, and not for high-risk automated decisions.
- QOS-UQCI / Quafu backend compatibility: Stage 10A Level 3 offline job specs, clean-room payloads, mock backend execution, result schemas, and optional upstream package boundaries. This is not production runtime or real hardware execution and does not access cloud services or tokens.
- Benchpress / Benchmarking compatibility: Stage 10B Level 3 clean-room local benchmark registry, runner, default suites, JSON / Markdown reports, and optional upstream Benchpress boundary. This is not official Benchpress output, not a full Benchpress replacement, and not production performance ranking.
- QuantumBridge Studio backend API: Stage 10C local service layer for catalog, workflow registry, input schema, execution, result store, export, benchmark, warnings/provenance, and REST-like local router. This is not a frontend UI, not a production API server, and does not access cloud services, tokens, or hardware.
- QuantumBridge Studio frontend prototype: Stage 10D static local prototype for catalog, workflow registry, workflow detail, mock local execution, results, warnings, provenance, benchmarks, and exports over Stage 10C seed data. This is not a production UI, not a production API server, and does not access cloud services, tokens, or hardware.
- QuantumBridge Studio backend/frontend integration: Stage 10E hardens the backend registry to frontend seed contract, static frontend client, representative sample results, benchmark/export samples, and local CLI. This is not a production UI or server and does not access cloud services, tokens, or hardware.

Current PennyLane ecosystem status:

- Operations, measurements, QNode, devices, qchem, transforms, gradients, templates, resources, and plugin-adjacent APIs: Level 0 inventory and Level 1 scaffold when PennyLane is installed.
- Existing P1 PennyLane tape/observable support remains a small reviewed Level 2 subset.
- PennyLane-Qiskit bridge: Stage 9H Level 3 educational bidirectional executable slice for H/X/Y/Z/RX/RY/RZ/Phase/CNOT/CZ/SWAP-style workflows, result schemas, warnings, provenance, and optional upstream plugin metadata.
- Full plugin ecosystem, Catalyst bridge, TensorFlow interface parity, and production qchem workflows are not implemented.

## Stage 7.2 Full Function Coverage

Stage 7.2 expands installed-environment verification and adapter schemas. It remains an ecosystem coverage project, not a full replacement for any upstream SDK.

| Ecosystem | Current coverage | Install extra | Status |
| --- | --- | --- | --- |
| Qiskit Nature | Level 0/1 plus ChemistryResult Level 2 wrapper and Stage 9D educational native H2 / LiH exact-diagonalization subset | `.[qiskit-nature]` | Verified: 0.8.0; native subset does not require upstream |
| Qiskit Finance | Level 0/1 plus FinanceResult Level 2 wrapper and Stage 9A educational portfolio native subset | `.[qiskit-finance]` | Verified: 0.4.1; not production finance |
| Qiskit Algorithms | Level 0/1 plus AlgorithmsResult Level 2 wrapper and Stage 9C educational native VQE/QAOA/Grover subset | `.[qiskit-algorithms]` | Verified: 0.4.0; native subset does not require upstream |
| Qiskit Machine Learning | Level 0/1 plus MLResult Level 2 wrapper and Stage 9E educational native quantum kernel / kernel classifier / QNN classifier subset | `.[qiskit-machine-learning]` | Verified: 0.9.0; native subset does not require upstream; not production ML |
| Qiskit Optimization | Level 0/1 plus OptimizationResult Level 2 wrapper and Stage 9B educational native QuadraticProgram subset | `.[qiskit-optimization]` | Verified: 0.7.0; native subset does not require upstream |
| Qiskit Dynamics | Level 3 educational offline one-qubit dynamics plus optional upstream boundary | `.[qiskit-dynamics]` | Z precession, Rabi drive, dephasing metadata; not production dynamics |
| Qiskit Experiments | Level 3 educational offline experiments plus optional upstream boundary | `.[qiskit-experiments]` | Rabi, T1, Ramsey synthetic workflows; not hardware calibration |
| Qiskit Metal | Level 0 inventory plus MetalDesignResult schema | `.[qiskit-metal]` | Install failed on Python 3.12; advisory/unsupported |
| Qiskit Aer | Level 0/1 plus AerResult Level 2 wrapper and Stage 9F educational native statevector / qasm counts / simple noise subset | `.[qiskit-aer]` | Verified: 0.17.2; native subset does not require upstream; no Aer parity claim |
| Mitiq / error mitigation | Stage 9G educational native ZNE and readout mitigation plus optional upstream Mitiq passthrough metadata | install upstream `mitiq` separately if needed | Native subset does not require upstream; not production error mitigation; no hardware calibration parity |
| PennyLane-Qiskit bridge | Stage 9H educational native bidirectional bridge and Bell equivalence proof plus optional upstream plugin metadata | install upstream `pennylane-qiskit` separately if needed | Native subset does not require upstream; not full plugin parity; no cloud/token/hardware |
| MQT Core / DDSIM / QMAP | Stage 9J educational native MQT Core-like circuit dict/QASM subset, DDSIM-like statevector/counts, and QMAP-like routing | install upstream MQT packages separately if needed | Native subset does not require upstream; not full MQT replacement; no decision-diagram or optimal-mapper parity |
| TorchQuantum / PyTorch-style QML | Stage 9K educational native TorchQuantum-like layer, tensor/batch forward, and classifier training | torch is optional for tensor interop; TorchQuantum is optional for upstream boundary metadata | Native subset does not require TorchQuantum; not full TorchQuantum/PyTorch replacement; not production QML or high-risk ML |
| QOS-UQCI / Quafu backends | Stage 10A offline QOS-UQCI job spec, DeviceSpec / CalSet / Manifest, Quafu-compatible payload, and mock execution | QOS-UQCI and pyquafu are optional upstream boundaries | No production runtime, no cloud/token/hardware access, no official endorsement |
| Benchpress / Benchmarking | Stage 10B local benchmark registry, runner, default suites, and JSON / Markdown reports | Benchpress is optional for upstream boundary metadata | Not official Benchpress output, not production performance ranking, no cloud/token/hardware access |
| QuantumBridge Studio Backend API | Stage 10C local catalog, workflow, execution, result, export, benchmark, and local router services | No heavy dependency; optional FastAPI boundary only | Backend-only, no frontend UI, no production server, no cloud/token/hardware access |
| QuantumBridge Studio Frontend Prototype | Stage 10D static local UI over generated Stage 10C seed data | No Node dependency required | Prototype-only, no production UI, no server, no cloud/token/hardware access |
| QuantumBridge Studio Integration | Stage 10E backend/frontend seed contract, 36 workflow details, 10+ sample results, benchmark/export samples, and local CLI | No heavy dependency; no Node install required | Local prototype integration only, no production UI/server, no cloud/token/hardware access |
| PennyLane full | Level 0/1 plus PennyLaneResult Level 2 wrapper | `.[pennylane-full]` | Verified: 0.42.3; no complete replacement claim |

Level 2 here means a QuantumBridge result-schema wrapper. It does not mean complete input conversion, behavioral parity, performance parity, or production equivalence.

Stage 9A adds `quantumbridge.compat.qiskit_finance.run_portfolio_optimization_native()` for a deterministic, educational four-asset mean-variance portfolio example and optional upstream calls through installed Qiskit Finance / Qiskit Optimization / Qiskit Algorithms packages. This is not investment advice, production portfolio optimization, real market-data support, or a complete Qiskit Finance replacement.

Stage 9B adds a Qiskit Optimization `QuadraticProgram` executable adapter:

- `quantumbridge.compat.qiskit_optimization.create_quadratic_program_native()`
- binary variables, linear/quadratic objectives, and linear constraints;
- deterministic brute-force exact solving for small educational problems;
- QUBO and Ising metadata generation;
- optional upstream exact passthrough when `qiskit-optimization` and `qiskit-algorithms` are installed.

This is not production optimization software, not a complete Qiskit Optimization replacement, and not an IBM or Qiskit endorsement.

Stage 9C adds Qiskit Algorithms executable adapters:

- `quantumbridge.compat.qiskit_algorithms.run_vqe_native()`
- `quantumbridge.compat.qiskit_algorithms.run_qaoa_native_maxcut()`
- `quantumbridge.compat.qiskit_algorithms.run_grover_native()`
- optional local upstream smoke paths through `run_vqe_upstream()`, `run_qaoa_upstream()`, and `run_grover_upstream()` when `qiskit-algorithms` is installed;
- `quantumbridge.schema.algorithms_results` result envelopes for native, upstream, and comparison results.

This is not production algorithm software, not a complete Qiskit Algorithms replacement, and not an IBM or Qiskit endorsement.

Stage 9D adds Qiskit Nature executable chemistry adapters:

- `quantumbridge.compat.qiskit_nature.run_h2_native()`
- `quantumbridge.compat.qiskit_nature.run_lih_native()`
- `quantumbridge.compat.qiskit_nature.build_h2_problem()`
- `quantumbridge.compat.qiskit_nature.build_lih_problem()`
- optional local upstream smoke paths through `run_h2_upstream_passthrough()` and `run_lih_upstream_passthrough()` when Qiskit Nature and its local chemistry stack are installed;
- `quantumbridge.schema.chemistry_results` result envelopes for molecular problems, qubit Hamiltonians, exact diagonalization, H2, LiH, upstream passthrough, and comparison results.

This is an educational exact-diagonalization chemistry slice, not production quantum chemistry, not a complete Qiskit Nature replacement, not a materials band-gap workflow, and not an IBM or Qiskit endorsement.

Stage 9E adds Qiskit Machine Learning executable educational adapters:

- `quantumbridge.compat.qiskit_machine_learning.run_quantum_kernel_native()`
- `quantumbridge.compat.qiskit_machine_learning.run_kernel_classifier_native()`
- `quantumbridge.compat.qiskit_machine_learning.run_qnn_classifier_native()`
- `quantumbridge.compat.qiskit_machine_learning.qnn_forward_native()`
- deterministic toy datasets, angle feature maps, state-fidelity kernel matrices, nearest-kernel classifier, and deterministic QNN grid-search classifier;
- optional local upstream introspection paths when `qiskit-machine-learning` is installed;
- `quantumbridge.schema.ml_results` result envelopes for native kernel, classifier, QNN, upstream passthrough, and comparison results.

This is an educational QML slice, not production machine learning, not a complete Qiskit Machine Learning replacement, not suitable for medical, financial, employment, identity, safety, or other high-risk automated decisions, and not an IBM or Qiskit endorsement.

Stage 9F adds Qiskit Aer-style executable educational simulator adapters:

- `quantumbridge.compat.qiskit_aer.run_statevector_simulator_native()`
- `quantumbridge.compat.qiskit_aer.run_qasm_simulator_native()`
- `quantumbridge.compat.qiskit_aer.run_noisy_qasm_simulator_native()`
- `quantumbridge.compat.qiskit_aer.execute_basic_circuit_native()`
- deterministic small-circuit statevectors, seeded qasm-style counts, and simple educational measurement bit-flip noise;
- optional local upstream passthrough paths when `qiskit-aer` is installed;
- `quantumbridge.schema.aer_results` result envelopes for native, noisy, upstream, and comparison results.

This is an educational simulator slice, not production simulator software, not a complete Qiskit Aer replacement, not Qiskit Aer noise-model parity, and not an IBM or Qiskit endorsement.

Stage 9G adds Mitiq-style executable educational error mitigation adapters:

- `quantumbridge.compat.mitiq.run_zne_native()`
- `quantumbridge.compat.mitiq.run_noisy_expectation_native()`
- `quantumbridge.compat.mitiq.linear_zero_noise_extrapolate()`
- `quantumbridge.compat.mitiq.run_readout_mitigation_native()`
- `quantumbridge.compat.mitiq.mitigate_readout_counts()`
- optional local upstream Mitiq dependency metadata when `mitiq` is installed;
- `quantumbridge.schema.error_mitigation_results` result envelopes for native ZNE, native readout mitigation, upstream passthrough, and comparison results.

This is an educational error mitigation slice, not production error mitigation, not a complete Mitiq replacement, not hardware calibration parity, and not an IBM, Qiskit, or Mitiq endorsement.

Stage 9H adds PennyLane-Qiskit bridge executable educational adapters:

- `quantumbridge.compat.pennylane_qiskit.run_qiskit_to_pennylane_bridge()`
- `quantumbridge.compat.pennylane_qiskit.run_pennylane_to_qiskit_bridge()`
- `quantumbridge.compat.pennylane_qiskit.run_bidirectional_bridge_equivalence()`
- Qiskit `QuantumCircuit` to QuantumBridge IR to PennyLane executable spec for a basic gate subset;
- PennyLane operation/tape/QNode metadata to QuantumBridge IR to Qiskit `QuantumCircuit`;
- Bell-state bidirectional equivalence proof over the native simulator;
- optional local upstream PennyLane-Qiskit plugin metadata when `pennylane-qiskit` is installed;
- `quantumbridge.schema.pennylane_qiskit_bridge_results` result envelopes for native bridge, upstream passthrough, IR summary, execution, and equivalence results.

This is an educational bridge slice, not a complete PennyLane-Qiskit plugin replacement, not full Qiskit or PennyLane parity, not production parity, and not an IBM, Qiskit, PennyLane, Xanadu, or PennyLane-Qiskit endorsement.

## IBM Quantum Ecosystem Clean-Room Parity

QuantumBridge's long-term ecosystem goal is clean-room functional parity for
the capability areas represented across IBM Quantum Ecosystem project listings:
project discovery, capability cataloging, optional adapters, executable
examples, result schemas, provenance, warnings, tests, and future Studio
visualization readiness.

This does not mean copying IBM's website, brand presentation, text, images,
icons, page layout, source code, or third-party project implementations.
Project names are used only to identify compatibility targets. The planning
baseline lives in `docs/roadmap/ibm_quantum_ecosystem_clean_room_parity_master_plan.md`.

## Stage 8D Qiskit Adapter Contract Hardening

Stage 8D adds a uniform contract across the Qiskit ecosystem lanes:

- capability metadata: `capability_level`, `production_ready`, `native_implementation`, and `upstream_required`;
- dependency and version reporting;
- generated public API inventory;
- public-object lookup, passthrough call/class helpers, result wrapping, schema conversion, warnings, provenance, unsupported metadata, and environment validation;
- Qiskit result schema envelopes for core, Aer, Nature, Algorithms, Finance, Optimization, Machine Learning, Dynamics, Experiments, Metal, Runtime, Addons, and conversions.

Generated Qiskit inventory snapshots live in `docs/compat/inventory/`, with summary matrices in `docs/compat/matrix/`. These snapshots are public-name inventories and do not copy upstream source or documentation prose.

Install one lane per environment:

```bash
python -m pip install -c requirements/constraints-qiskit-nature.txt -e '.[qiskit-nature]'
python -m pip install -c requirements/constraints-qiskit-algorithms.txt -e '.[qiskit-algorithms]'
python -m pip install -c requirements/constraints-qiskit-dynamics.txt -e '.[qiskit-dynamics]'
python -m pip install -c requirements/constraints-qiskit-metal.txt -e '.[qiskit-metal]'
```

Use `python scripts/verify_ecosystem_installed_envs.py` to create isolated temporary verification environments. Do not install every optional extra together. Qiskit Runtime verification is offline-only: QuantumBridge does not request tokens or contact IBM Cloud. Finance, chemistry, ML, Experiments, Dynamics, and Metal coverage is experimental and not production-grade. Metal coverage does not imply chip fabrication readiness or external electromagnetic solver validation.

### Qiskit Experiments / Dynamics Offline Slices

Stage 9I adds executable educational offline workflows for:

- Rabi, T1, and Ramsey synthetic experiments with deterministic fitting metadata.
- Single-qubit Z precession, Rabi drive dynamics, and dephasing metadata simulation.
- Optional upstream `qiskit-experiments` and `qiskit-dynamics` passthrough wrappers when those packages are installed.

These paths do not access IBM Runtime, cloud services, tokens, or real hardware.
They are not hardware calibration, production experiment analysis, production
dynamics software, or full replacements for upstream projects.

### MQT Core / DDSIM / QMAP Compatibility Slice

Stage 9J adds clean-room educational workflows for:

- MQT Core-like circuit dictionaries, QuantumBridge IR roundtrip, and a small QASM subset artifact.
- DDSIM-like statevector and counts simulation for small circuits using QuantumBridge native simulation.
- Decision-diagram-inspired metadata such as support bitstrings, unique amplitude counts, and compression hints.
- QMAP-like line/ring/full/custom topology validation, greedy CNOT routing, SWAP insertion, mapping cost, and original-vs-mapped execution comparison.
- Optional upstream MQT dependency boundary helpers when local MQT packages are installed separately.

These paths do not access cloud services, tokens, or real hardware. They are
not full MQT Core, DDSIM, or QMAP replacements, not decision-diagram parity,
not optimal mapping software, and not production compiler/simulator/mapper
parity.

### TorchQuantum / PyTorch-style QML Compatibility Slice

Stage 9K adds clean-room educational TorchQuantum-like QML workflows:

- tensor-like adapters for Python lists, NumPy arrays, and optional torch tensors;
- native small-circuit quantum layer forward passes over QuantumBridge simulation;
- batch forward execution with probabilities, outputs, predictions, warnings, and provenance;
- deterministic toy classifier training with grid search and explicit high-risk ML warnings;
- optional upstream TorchQuantum passthrough boundary when installed separately.

This slice is not a full TorchQuantum replacement, not a full PyTorch
replacement, not production QML training, and not suitable for medical,
financial, employment, identity, safety, or other high-risk automated
decisions. It does not access cloud services, tokens, or real hardware.

### QOS-UQCI / Quafu Backend Compatibility Slice

Stage 10A adds clean-room offline backend execution workflows:

- QuantumBridge IR to QOS-UQCI clean-room IR and job spec;
- QOS-UQCI DeviceSpec, CalSet, Manifest, and OpenQASM compatibility artifact;
- QOS-UQCI offline mock runtime returning QuantumBridge result schemas;
- QuantumBridge IR to Quafu-compatible payload and job spec;
- Quafu offline mock backend returning QuantumBridge result schemas;
- optional QOS-UQCI and pyquafu upstream boundary metadata.

This slice is not production QOS runtime support, not production Quafu backend
support, not real hardware execution, and not official endorsement. It does not
read tokens or access cloud services by default.

### Benchpress / Benchmarking Compatibility Slice

Stage 10B adds clean-room local benchmark execution workflows:

- benchmark case registry and default local suites;
- basic circuit, simulator, algorithms, finance / optimization, chemistry,
  QML, mitigation, backend, and bridge benchmark cases;
- deterministic local runner with pass/fail metrics;
- JSON and Markdown report generation;
- optional upstream Benchpress boundary metadata.

This slice is not a full Benchpress replacement, not official benchmark output,
not a production performance ranking system, and does not access cloud services,
tokens, or real hardware.

### QuantumBridge Studio Backend API Slice

Stage 10C adds a local backend API executable slice for future QuantumBridge
Studio frontends:

- catalog API over QuantumBridge-owned ecosystem metadata;
- workflow registry and input schema API for executable slices;
- local execution service and in-memory result store;
- JSON, Markdown, Python snippet, and notebook-stub exports;
- benchmark service wrapping the Stage 10B local benchmark runner;
- REST-like local router without starting an HTTP server;
- optional FastAPI adapter boundary when FastAPI is installed separately.

This slice is backend-only. It is not a frontend UI, not a production API
server, not an official IBM / Qiskit / PennyLane / Benchpress service, and does
not access cloud services, tokens, credentials, or real hardware.

### QuantumBridge Studio Frontend Prototype

Stage 10D adds a static local frontend prototype under `studio/`:

- ecosystem catalog view;
- workflow registry view;
- workflow detail view with input/output schema;
- mock local execution panel;
- result JSON, warning, and provenance viewers;
- benchmark report panel;
- JSON, Markdown, Python snippet, and notebook-stub export display.

Generate local seed data and run the frontend smoke check:

```bash
python3 examples/studio_generate_frontend_seed_data_quantumbridge.py
node studio/scripts/smoke-check.mjs
```

The prototype uses generated Stage 10C seed data and does not start a server,
open ports, access cloud services, read credentials, access real hardware, copy
third-party UI/branding/prose, or claim production UI readiness.

### QuantumBridge Studio Backend/Frontend Integration

Stage 10E hardens the local Studio loop:

- backend workflow registry is the canonical frontend data source;
- seed generation writes catalog, workflows, workflow details, sample results,
  benchmark report, exports, and schema version metadata;
- frontend local clients expose catalog, workflows, workflow detail, input
  schema, sample results, mock execution, benchmark report, and exports through
  one response shape;
- `python -m quantumbridge.studio` provides local JSON commands for seed
  generation, workflow listing, local execution, benchmark, and export.

Stage 10E remains a local prototype. It does not claim production readiness or
official endorsement. It is not a full replacement and makes no production
parity claim.

## P1 Controlled Expansion

Stage 5 adds controlled P1 subset coverage:

- CI dependency matrix workflow.
- PennyLane installed-environment tests, skipped when the optional dependency is unavailable.
- OpenQASM 2.0 subset parser improvements: include handling, comments, multi-register flattening, barrier no-op, whole-register measurement.
- Additional gates: S, SDG, T, TDG, Phase, SWAP, experimental iSWAP, and basic CCX matrix path.
- Operator/Hamiltonian enhancements.
- Testable compiler PassManager basics.
- Basic noise channels and error model metadata.

## Not Supported Yet

- Full Qiskit feature parity.
- Full PennyLane feature parity.
- Full Qiskit Nature / Algorithms / Finance / Optimization / Machine Learning / Aer / Runtime / Experiments parity.
- Full MQT Core / DDSIM / QMAP parity.
- Full TorchQuantum or PyTorch parity, production QML, or high-risk ML decision support.
- Production QOS runtime, production Quafu backend, real quantum cloud access,
  token handling, or real hardware execution.
- Full Benchpress replacement, official benchmark claims, or production
  performance ranking.
- QuantumBridge Studio production UI, production API server claims, cloud
  service execution, token handling, or real hardware execution.
- Full OpenQASM grammar.
- Hardware cloud providers.
- Production Qiskit Aer/noise integration.
- Full transpiler and pass ecosystem.
- Full PennyLane plugin ecosystem.
- Production quantum chemistry, finance, optimization, or machine learning workflows.
- Materials band gap workflows.
- Production-grade visualization.
- Production release guarantees.

## Install

For local development:

```bash
python -m pip install -e .
```

Optional adapter extras:

```bash
python -m pip install -e '.[qiskit]'
python -m pip install -e '.[qiskit-core]'
python -m pip install -e '.[qiskit-aer]'
python -m pip install -e '.[qiskit-nature]'
python -m pip install -e '.[qiskit-finance]'
python -m pip install -e '.[qiskit-optimization]'
python -m pip install -e '.[qiskit-machine-learning]'
python -m pip install -e '.[pennylane]'
python -m pip install -e '.[pennylane-full]'
python -m pip install -e '.[torch]'
python -m pip install -e '.[jax]'
python -m pip install -e '.[dev]'
```

Qiskit, Qiskit ecosystem packages, PennyLane, Torch, and JAX are optional dependencies. Tests that require unavailable optional packages should skip with an explicit reason or run only dependency-smoke checks.

Do not commit third-party package source trees, local `site-packages`, downloaded wheels, or vendored Qiskit / PennyLane / PySCF / OpenFermion / Quafu code into this repository.

### Ecosystem dependency lanes

Stage 7 uses separate optional dependency lanes. Do not assume all optional dependencies can coexist in one environment.

- `qiskit-core`: Qiskit core public API inventory and selected P1 adapters.
- `qiskit-aer`: optional Aer simulator/noise passthrough plus Stage 9F educational native statevector, qasm-counts, and simple sampling-noise slice.
- `mitiq`: no default extra; if installed separately, QuantumBridge records optional upstream Mitiq passthrough metadata while native Stage 9G ZNE/readout examples run without it.
- `pennylane-qiskit`: no default extra; if installed separately, QuantumBridge records optional upstream plugin metadata while native Stage 9H bridge examples run without it.
- `mqt`: no default extra; if installed separately, QuantumBridge records optional upstream MQT Core/DDSIM/QMAP passthrough metadata while native Stage 9J examples run without it.
- `torchquantum`: no default extra; if installed separately, QuantumBridge records optional upstream TorchQuantum passthrough metadata while native Stage 9K examples run without it. `torch` is optional for tensor interop only.
- `qos_uqci` / `quafu`: no default extra; if installed separately,
  QuantumBridge records optional upstream boundary metadata while Stage 10A
  native examples run offline without cloud, token, or hardware access.
- `qiskit-finance`: optional finance application/data-provider/circuit inventory.
- `qiskit-optimization`: optional optimization inventory and passthrough scaffold.
- `qiskit-machine-learning`: optional QNN/kernel/classifier/Torch connector inventory.
- `pennylane-full`: optional PennyLane operations, measurements, QNode, devices, qchem, transforms, gradients, templates, resources, and plugin-adjacent inventory.
- `ecosystem-chemistry`: optional Qiskit Nature / Algorithms / PySCF / OpenFermion lane; not production chemistry.

If environment conflicts appear, use separate virtual environments such as `quantumbridge-ecosystem-qiskit`, `quantumbridge-chemistry`, and `quantumbridge-pennylane`. Materials band gap workflows are not implemented.

## Test

Run:

```bash
pytest -q
```

Run with skip reasons:

```bash
pytest -q -rs
```

In the current P1 hardening environment after installing the PennyLane extra, the acceptance run is:

```text
68 passed
```

Core-only style tests:

```bash
pytest -q -rs tests/information tests/operators tests/primitives tests/compiler tests/noise tests/qml/test_qml_gradient_extended.py
```

Qiskit extra tests:

```bash
pytest -q -rs tests/compat/test_qiskit_adapter_realistic.py tests/compat/test_qiskit_import_basic_circuit.py tests/compat/test_qiskit_export_basic_circuit.py tests/compat/test_qiskit_roundtrip_ir.py
```

PennyLane extra tests:

```bash
pytest -q -rs tests/compat/test_pennylane_installed_environment.py tests/compat/test_pennylane_adapter_realistic.py tests/qml/test_pennylane_observable_bridge.py tests/qml/test_pennylane_qnode_basic.py
```

Coverage:

```bash
pytest --cov=quantumbridge
```

Only run coverage if `pytest-cov` is installed. In the Stage 5.5 local gate, the dev extra installed `pytest-cov` successfully and coverage passed.

Local matrix simulation:

```bash
bash scripts/run_local_matrix.sh
```

Ecosystem inventory scripts:

```bash
python scripts/inventory_qiskit_core_api.py
python scripts/inventory_qiskit_aer_api.py
python scripts/inventory_qiskit_finance_api.py
python scripts/inventory_qiskit_optimization_api.py
python scripts/inventory_qiskit_machine_learning_api.py
python scripts/inventory_pennylane_full_api.py
```

Suggested P1 RC tag name for a future explicit release action:

```text
v0.1.0-p1-rc1
```

Do not create or push the release tag until remote CI has run and the release is explicitly approved.

## Native Compiler Checkpoint

The native compiler now includes a wire-dependency DAG, validated Target/Layout,
directed shortest-path routing, exact-phase basis translation, scoped dynamic
body compilation and static ASAP/ALAP scheduling. QNode.compile preserves the
tested first-order native/Torch/Autograd derivatives. It is a tested subset,
not full Qiskit/PennyLane parity, optimal routing, hardware scheduling or real
Quafu verification. Studio remains deferred.

See [Stage 11L report](docs/implementation/stage11l_native_transpiler_report.md)
and the SDK-free `examples/native_transpile_*.py` workflows.

## Native Simulator Checkpoint

`quantumbridge.simulator.NativeAerSimulator` reuses native Backend/Job execution
with CPU vector, density, MPS, Clifford tableau and bounded unitary/superoperator
methods. Circuit Save/Set instructions execute at their actual positions;
`result.data()` returns labelled snapshots. Native `NoiseModel` supports
gate-attached Kraus errors and a bounded correlated-readout subset, not just
changes to terminal counts. This remains a tested subset, not full Aer parity.

```bash
python -m examples.native_aer_save_set
python -m examples.native_aer_gate_noise
python -m examples.native_aer_density_mps
python -m examples.native_aer_dynamic_snapshots
```

GPU, extended stabilizer, general tensor-network, unsupported options and
oversized saves explicitly fail. Native bitstrings remain wire/classical-zero
first; quantum-error matrices use little-endian qargs. SDK-native object
shapes/signatures are not universally interchangeable. See the
[Stage 11N report](docs/implementation/stage11n_native_aer_report.md).

## Native Templates Checkpoint

`quantumbridge.templates` provides eight concrete native templates: Angle,
Basis and AmplitudeEmbedding, Basic/StronglyEntanglingLayers, IQPEmbedding,
QFT and first/second-order TrotterProduct. Objects expose actual native
`decomposition()`, `to_circuit()` and `apply(tape_or_circuit, wire_map=...)`.
Angle/Basic/Strong inputs support the existing native first-order gradients
and optional Torch/Autograd interfaces without an upstream SDK executor.

```bash
python -m examples.native_angle_embedding_qnode
python -m examples.native_entangling_templates_training
python -m examples.native_qft_and_trotter
```

Important boundaries: functional `qml.angle_embedding` retains default Y;
`templates.AngleEmbedding` defaults to X. BasicEntanglerLayers now uses RX
and the public ring convention; the legacy RY/open-chain helper remains
`basic_entangler`/`hardware_efficient_ansatz`. Amplitudes are genuinely prepared
but bounded to dimension 64 by default, static, and require zero target input.
Their native IR is supported, while opaque matrix QASM/QPY exports reject.
Trotter approximates `exp(+i t H)`, not the opposite time-sign convention.
The old functional `basic_entangler_layers` entry emits a migration warning;
use `hardware_efficient_ansatz` to retain RY/open-chain behavior, or the
explicit `templates.BasicEntanglerLayers` object for RX/ring behavior.
This is not full PennyLane template parity. See the
[template report](docs/implementation/stage11o_ab_native_templates_report.md).

## Native Transform Execution Checkpoint

`quantumbridge.transforms.TransformProgram` now composes native
`Tape -> (Tapes, processing_fn)` functions. `execute_transform` runs each
child with the existing native QNode execution plan, then reconstructs
results in reverse transform order. `return_metadata=True` exposes actual
child evaluation IDs, operation counts, shot partitions and stage order.

```bash
python -m examples.native_transform_program
python -m examples.native_transform_qnode
python -m examples.native_measurement_splitting
python -m examples.native_qchem_transform_energy
python -m examples.native_qwc_shot_allocation
```

`split_hamiltonian` is an internal expval-to-term-Tapes prototype, **not**
PennyLane `split_to_single_terms` parity. It handles static real Hamiltonian
coefficients, identity/duplicate/zero terms and nested output trees.
`merge_rotations` reuses existing native algebra within safe segments and
retains symbolic parameter graphs. Native transforms now dispatch to Tape,
QNode or a native callable with explicit `qnode_options`. Both
`merge_rotations(node)` and `TransformProgram([...]).execute_qnode(node, x)`
build a new batch with current inputs, without changing the original QNode.

The actual `split_to_single_terms` contract instead emits **one** Tape with
distinct measurements: repeated native Pauli measurements are deduplicated,
identity coefficients are offsets, and nested results/shot partitions are
reconstructed. It is not QWC grouping. `cancel_inverses`, bounded native
`decompose` and `compile` also execute; compilation reuses the native target,
layout/routing engine but rejects unproved measurement remapping/ancilla growth.
Definitions are native Circuit/Instruction/Template objects, not SDK source.

The separate `split_non_commuting` transform now groups native Pauli
expvals/Pauli variances using QWC, disjoint `wires`, or no grouping. Default is
QWC for this all-Pauli subset. Generally commuting XX/YY do **not** share a
local-basis group. Each finite-shot group makes one native `Device.sample`
call and derives all its word eigenvalues from that exact joint bitstring
buffer. Metadata retains the basis, actual sampling IDs, bitstrings,
coefficients, means, covariance, allocated shots and source output mapping.

`shot_dist=None` gives every group the full shots; `"uniform"` and `"weighted"`
share a total budget, conserving each shot-vector partition independently.
Weighted allocation uses group coefficient L1 norms. Integer shares are
floored, with remainder shots assigned in group order. Insufficient effective
group samples reject; weighted zero-contribution groups are omitted.
Hamiltonian expectation uncertainty includes within-group covariance and
independent-group estimator variances. A single shot has unknown standard
error, not a fabricated zero error. This is not a full quantum Hamiltonian
variance across incompatible groups or a certified hardware error bound.

Public tests use PennyLane 0.44.1/0.42.3; the older version has no `shot_dist`
API. The newer version flattens distributed shot vectors and can omit groups
with zero allocated shots. Native partition preservation and explicit
undersampling rejection are documented differences, not full public parity.
General observables, local measurement overrides, sum variances, random/custom
allocation and complete transform APIs remain unsupported.

Other transforms retain their per-child/per-measurement shot semantics.
Recorded transformed QNodes now support first-order native Jacobians,
VJP/JVP and Torch/Autograd backward. Child quantum means use the existing
native shift/adjoint engine; a bounded native postprocessing Jacobian follows
the actual reconstruction order. Hamiltonian coefficient graphs survive
construction/copying, including a parameter shared by gate angles and
coefficients: both `c*dmean` and `dc*mean` contribute. Ordinary live-coefficient
QNodes reuse native single-term preprocessing, not a second execution engine.
Quantum primals are unboxed for native execution while the original classical
graphs remain registered. Passive covariance/SE/bitstrings/UUIDs are not
differentiable leaves. These are explicit bounded Jacobian products, not a
matrix-free VJP or full derivative-device protocol.

Finite-shot shift evaluations really resample each group with independent
shift/partition streams and report actual forward/derivative physical work.
Sampled gradients are stochastic estimates; energy SE is not a gradient error
bound. Finite-shot transformed variance gradients, trainable zero-weight
weighted allocation, fixed local seed gradients, raw live Tape execution,
postselection/reused parameterized loops, higher order, capture, complete JAX/TF and
GPU contracts remain explicitly unsupported. A prepared coefficient-bearing
plan rejects changed coefficient primals; QNode calls construct fresh plans.

```bash
python -m examples.native_transformed_autodiff
python -m examples.native_mps_template_transform_gradient
python -m examples.native_qchem_transformed_gradient
```

The 100-wire GHZ/template/QWC native shift proof forbids dense expansion and
records real MPS bond/storage/truncation telemetry without extra resource-only
executions. It does not imply arbitrary 100-wire efficiency. Full Stage 11O-B
and full SDK parity remain incomplete. See the
[B2/B3 report](docs/implementation/stage11o_b2_b3_native_transform_report.md).
Actual grouped sampling and its boundaries are in the
[B4 report](docs/implementation/stage11o_b4_native_qwc_shot_allocation_report.md).
Gradient execution and remaining limits are in the
[B5 report](docs/implementation/stage11o_b5_native_transformed_autodiff_report.md).

## Native Coherent Deferred Measurements

`transforms.defer_measurements` turns the supported native Z-measurement /
conditional-unitary subset into an actual larger unitary circuit. Every
measurement owns a fresh zero-initialized record ancilla, including when its
original wire is subsequently changed. Logical probabilities and reduced
density outputs exclude unrequested ancillas. This is not continued dynamic
execution hidden behind a transform name.

```python
from quantumbridge.qml import QNode
from quantumbridge.transforms import defer_measurements
from quantumbridge.utils.math import PauliZ

def circuit(tape, theta):
    tape.ry(theta, 0)
    m = tape.measure(0)
    tape.h(0)
    tape.cond(m, lambda branch: branch.x(1))
    return tape.expval(PauliZ(1))

node = defer_measurements(QNode(circuit, 2))
energy = node(0.37)
derivative = node.gradient(0.37)
```

Supported terminal `probs`, `sample` and `counts` accept a native `records=`
keyword. Records precede explicitly selected quantum wires in their bitstrings;
omitting wires with records requests records only. They are actual classical
records on the original dynamic engine and actual ancilla reads after defer,
not the changed state of the originally measured wire. This native interface
does not claim the complete upstream MeasurementValue protocol.

The transform composes with native QWC grouping, shot allocation, compilation,
first-order Jacobian/VJP and Torch/Autograd. Finite-shot derivatives retain
their stochastic semantics and actual physical-shot receipts. Capacity and
expansion limits apply before constructing oversized controlled matrices.
Device/Target capacity must include extra wires; simulator success is not
hardware validation. A 100-logical-wire GHZ/template/feedback/gradient proof
uses 101 real MPS wires with full dense expansion forbidden.

Reset, postselection, noisy feedback, conditional measurements, runtime loops,
classical Store/arithmetic, mid-state snapshots, undefined full state output
and unproved opaque/observable mappings remain explicitly unsupported.
The complete SDK functional parity goal is still incomplete.

```bash
python -m examples.native_defer_measurements
python -m examples.native_defer_reused_wire
python -m examples.native_defer_transform_gradient
python -m examples.native_defer_mps_100qubits
python -m examples.native_stage11o_integrated_h2
```

See the [B6 report](docs/implementation/stage11o_b6_native_deferred_measurements_report.md),
[joint acceptance](docs/implementation/stage11o_joint_acceptance_report.md)
and [remaining gaps](docs/implementation/stage11o_remaining_functional_parity_gaps.md).

## Native JAX Autodiff

Optional `interface="jax"` now connects the existing native QNode and
TransformProgram to real first-order `jax.grad`, `value_and_grad`, `jacrev`,
`jacfwd`, `jvp` and VJP workflows. JAX retains preprocessing/outer-loss graphs;
QuantumBridge supplies actual native shift/adjoint and child/postprocessing
Jacobians, including shared gate/Hamiltonian coefficient product rules.
This is not PennyLane execution, finite-difference fallback or simulator backprop.

```python
import jax
import jax.numpy as jnp
from quantumbridge import PauliZ
from quantumbridge.qml import QNode

def circuit(tape, theta):
    tape.ry(jnp.sin(theta) + 2 * theta, 0)
    return tape.expval(PauliZ(0))

node = QNode(circuit, 1, interface="jax", diff_method="parameter-shift")
value, gradient = jax.value_and_grad(node)(jnp.asarray(0.37))
```

Analytic and explicit-immutable-integer-seeded sampling support tested CPU
JIT/VMAP subsets via public pure host callbacks and declared shape/dtype.
VMAP is sequential CPU host execution, not GPU quantum batching or quantum
simulation compiled into XLA. Native plans/devices are fresh per host call;
dynamic PRNG keys, complete plugin protocols and arbitrary dynamic shapes
remain unsupported. Float32/float64 respect the caller's x64 configuration;
QuantumBridge never changes that process setting on import.

`node.jax_with_status(...)` returns `(value, status)` for numerical runtime
validation: 0 success, 1 invalid/nonfinite value, 2 unsupported runtime path,
3 budget failure, 4 other runtime failure. Failures return NaN value/gradient
data, not intentional exceptions inside a pure callback. Status has zero
tangent. Static shape/capability errors reject before callback creation.
Ordinary eager calls retain normal exceptions.

`JAXBridgeLimits` bounds inputs, source/expanded parameters, output/Jacobian
storage, transform Tapes, sampled work and mapped batch expansion. Its callback
budget bounds logical mapped expansion per axis, not the actual optimizer-
scheduled count: JIT can drop, duplicate or reorder pure callbacks. JIT metadata
is explicitly a trace-configuration description, never an actual execution
count or resource receipt. Eager native receipts remain auditable.
Finite gradients are estimates, with independent shifted/group/partition
streams; forward SE is not a strict gradient-error bound.

Opt-in bounded second derivatives are described below. Third/higher orders,
capture/Catalyst, GPU/TPU, native backprop, full derivative-device protocols
and complete SDK parity remain open.
Public comparisons document version-specific differences rather than treating
upstream unsupported paths as positive matching evidence.

```bash
JAX_ENABLE_X64=1 python -m examples.native_jax_qnode_gradient
JAX_ENABLE_X64=1 python -m examples.native_jax_transform_training
JAX_ENABLE_X64=1 python -m examples.native_jax_mps_100qubits
JAX_ENABLE_X64=1 python -m examples.native_jax_qchem_h2
```

See the [JAX audit](docs/implementation/stage11p_a_jax_gap_audit.md) and
[JAX implementation report](docs/implementation/stage11p_a_native_jax_autodiff_report.md).

## Native TensorFlow Autodiff

Optional `interface="tf"` / `"tensorflow"` now uses real `GradientTape`,
`tf.Variable`, float32/float64 and first-order native quantum VJPs. TensorFlow
owns classical preprocessing and outer losses; QuantumBridge's existing
shift/adjoint, Device and transformed child/postprocessing Jacobians own
quantum execution. No PennyLane/TF Quantum quantum engine is used.

```python
import tensorflow as tf
from quantumbridge import PauliZ
from quantumbridge.qml import QNode

def circuit(tape, theta):
    tape.ry(tf.sin(theta) + 2 * theta, 0)
    return tape.expval(PauliZ(0))

node = QNode(circuit, 1, interface="tf", diff_method="parameter-shift")
theta = tf.Variable(0.37, dtype=tf.float64)
with tf.GradientTape() as recording:
    loss = node(theta) ** 2
gradient = recording.gradient(loss, theta)
```

Install the optional `.[tensorflow]` extra. Supported static `tf.function`
graphs use same-process CPU host callbacks, not a quantum XLA/GPU lowering.
On the tested macOS host, importing PyArrow 24 before TensorFlow 2.21 reproduced
a native-library hang even without QuantumBridge. Validation uses an isolated
TF environment or a PyArrow 23.0.1 test overlay; arbitrary co-installability is
not guaranteed. Details and exact environment conditions are in the TF report.
Recording is symbolically traced even for eager calls, preventing `float(x)` /
`x.numpy()` from silently detaching trainable gate expressions. Variables,
including tested closures, are read into explicit packed tensor inputs outside
the custom-gradient quantum operation. Fully defined static shapes and real
expval/variance/probability output trees are required. Native templates,
compile/merge/defer/QWC, live Hamiltonian coefficients and multiple child
Tapes preserve their tested chain rules.

`tensorflow_with_status` returns forward value/status (0 success, 1 invalid,
2 unsupported, 3 budget, 4 unexpected failure); failed derivatives are NaN.
Ordinary graph calls assert on callback failure. Static unsupported contracts
raise before callback creation. `TensorFlowBridgeLimits` bounds input/output,
source/expanded parameters, Jacobians, child Tapes and forward-plus-backward
shot work. `tensorflow_batch(values)` explicitly bounds single-argument static
leading-axis mapping. Callback bounds describe this API's logical expansion,
not arbitrary external TF scheduling, mapping or repeated graph invocations.
Eager metadata reports actual forward/backward work; graph metadata explicitly
contains trace configuration only, never invented execution counts.

Finite-shot gradients require an explicit immutable integer seed and are
estimates, not pathwise derivatives or universal accuracy guarantees. Separate
forward work, native base/linear/shift work, partitions and group/shift seeds
are receipted. Opt-in second derivatives are described below; third/higher
orders, arbitrary dynamic shapes/control flow,
complex/full-state derivatives, distributed execution, portable SavedModel,
XLA, GPU/TPU and simulator backprop remain unsupported/unverified. Full
TensorFlow/PennyLane/Qiskit parity remains incomplete.

```bash
python -m examples.native_tensorflow_qnode_gradient
python -m examples.native_tensorflow_transform_training
python -m examples.native_tensorflow_mps_100qubits
python -m examples.native_tensorflow_qchem_h2
```

These are real gradient, twenty-update SGD, non-dense low-entanglement 100-wire
MPS and fixed-geometry native Gaussian/RHF/JW H2 workflows, not general MPS
efficiency, differentiable Hartree-Fock or ground-state convergence guarantees.
See the [TF audit](docs/implementation/stage11p_b_tensorflow_gap_audit.md) and
[TF report](docs/implementation/stage11p_b_native_tensorflow_autodiff_report.md).

## Native Second Derivatives

Set `max_diff=2` for bounded native Hessians, streamed Hessian-vector products
and actual CPU Torch/Autograd/JAX/TF second-order graphs. The default remains
first order; third derivatives explicitly reject. Quantum second derivatives
use convolved exact parameter-shift recipes, including the two frequencies of
controlled rotations, not finite differences or an upstream quantum engine.
`best` still selects parameter-shift for second order. Explicit `adjoint` now
uses native tangent-over-reverse second derivatives on analytic, noiseless,
unitary StatevectorDevice plans. Sampled, mixed/MPS and general dynamic second
adjoint paths remain unsupported; simulator backprop is not implemented.

```python
import numpy as np
from quantumbridge import PauliZ
from quantumbridge.qml import QNode

def circuit(tape, x):
    tape.ry(x[0] * x[1], 0)
    return tape.expval(PauliZ(0))

node = QNode(circuit, 1, max_diff=2)
hessian = node.compute_hessian(np.array([0.7, 0.2]))
product = node.compute_hvp(np.array([0.4, -0.3]), np.array([0.7, 0.2]))
```

`execute_and_compute_hessian` and `execute_and_compute_hvp` also return the
executed value. For multiple selected arguments, Hessians are argument-pair
blocks containing the output tree; HVPs contain one output tree per argument.
Native and CPU Torch/Autograd/JAX/TF HVPs accumulate directional second
contributions without allocating an output-by-parameter-by-parameter Hessian.
JAX has explicit linear tangent, transpose, CPU lowering and mapped rules;
reverse-mode output rows are preflighted together and executed sequentially.
First Jacobians and classical maps can still be dense. This is not device-native
quantum differentiation or an assertion that all quadratic memory is removed.
Fixed numeric source angles are not treated as framework variables.

Actual transformed children and native processors retain nonlinear input,
gate/coefficient and cross-child product rules. Analytic variances include
second-moment and mean-gradient outer-product corrections. Explicitly seeded
finite-shot linear expectations/probabilities retain partition/group streams
and physical-work receipts. Direction-independent seeded shift schedules retain
HVP linearity, including stationary input maps; fixed numerical angles do not
acquire derivative recipes. Sampled variance, unverified nonlinear sampled
postprocessing and parameter-dependent weighted integer shot allocation reject.
Postselection, reused parameterized loops and general measurement-record second
derivatives remain unsupported. Supported coherent deferred analytic children
are tested separately.

`quantumbridge.diff.HigherOrderLimits` bounds derivative order, input/output,
Hessian storage, shifted circuits, tapes, shots, trace/processor nodes and
derivative-buffer memory. Native devices retain their separate state-memory
limits; this is not a universal process-memory or arbitrary-user-code limit.
Second adjoint additionally preflights a conservative combined bound for its
working state vectors and derivative buffers against both memory limits, plus
`max_adjoint_operations` dense-element work. It replays one forward tangent and
one reverse curvature sweep per output/direction without retaining the entire
state trajectory. This path does not allocate quantum gate-pair Hessians or
substitute shifted circuits. Source graph ownership and arbitrary user converter
internals are outside the combined working-buffer claim.
Non-dense 100-wire MPS and fixed-geometry H2 ansatz tests do not imply arbitrary
MPS efficiency, geometry Hessians, differentiable Hartree-Fock or full parity.

Run `python -m examples.native_qnode_second_order` or
`python -m examples.native_framework_hvp --interface all --chemistry`.
The latter reuses one native H2 Hamiltonian build across all four frameworks.
Add `--method adjoint` to run the same workflow through second adjoint.
Current receipts and remaining gaps:
[second-adjoint report](docs/implementation/stage11p_c2a_native_second_adjoint_report.md).

### Derivative Execution Work In Progress

Stage 11P-C3 is adding native Device derivative requests, shared resource
reservations and explicit CPU JAX/TF derivative-status queries. Current tested
slices do not complete Device parity, graph-wide accounting or strict process
memory bounds. The full functional replacement goal remains incomplete.
See the [resource progress report](docs/implementation/stage11p_c3_unified_resource_report.md)
and [status progress report](docs/implementation/stage11p_c3_derivative_status_report.md).

C3-R1 is correcting native matrix preallocation and retained Torch/Autograd
array ownership. Graph snapshots and returned native buffers can remain charged
after a callback returns; views share a storage lease and real copies do not.
Public Device derivative Results now retain their array leases too; status
errors omit traceback references to failed native temporaries. Known Tape and
host-device ndarray copies are pre-reserved inside an active resource scope.
Native JAX/TF kernels now use a separate construction pool for known ndarray
snapshots, with per-request resident admission before quantum work. Classical
source-map tracing reuses the description without copying the snapshot again.
Persistent bytes, resident admission and new-allocation reservations are
separate quantities, not an RSS bound. Focused compiled-query tests cover
runtime rejection/reuse and graph lifetime; ordinary graph-wide budgets and
universal allocator ownership remain incomplete.
This is ongoing work, not universal memory safety or full SDK parity. See the
[ownership progress report](docs/implementation/stage11p_c3_r1_resource_ownership_report.md).

The ongoing R1 slice also tests ordinary TensorFlow gradient graph lifetimes
and retained ConcreteFunctions, rejects unbounded foreign persistent-copy
protocols before dispatch, and reserves strided flatten/reconstruction copies.
MPS workspace auditing and complete framework graph budgeting remain unfinished;
passing focused tests is not full Qiskit/PennyLane functional parity.

The MPS foundation now pre-reserves known tensor/contraction/QR/SVD/reduction
arrays and retains their actual roots, with tested gate-failure rollback and
raw Result/view lifetimes. Live MPS saves now retain native tensors through
Context/Backend/Job/Result; `result.matrix_product_state(label)` defensively
copies labeled native saves, preserving conditional/pershot nesting. Existing
`data()` and JSON retain serialized dict/list formats. JSON-loaded Results do
not claim restored native snapshot ownership. Completed MPS trajectories are
counted even if later measurement/Kraus/snapshot work fails; unknown internal
partial work remains unavailable. General serialization/observable scratch,
complete failure auditing and final R1 gates remain unfinished; external
LAPACK workspace is not measured RSS.

Scoped observable/IR conversions now reserve known dense outputs and numeric
dtype copies, and Hermitian checks use an explicit validation arena. Native
Pauli phase/wire order and the H2 QubitHamiltonian path are retained. Result
state/density and saved-array imports own their numerical roots inside an
active ResourceScope. Existing JSON codecs additionally bound UTF-8 transport,
nodes, depth, numeric elements and labels using `SerializationLimits`;
Result JSON methods accept `serialization_limits=`. Defaults include 16 MiB
transport, JSON depth 256, 100000 nodes, 1048576 numeric elements and 1024 saved labels.
These are not strict Python heap/RSS bounds. Opaque managed matrix factories
remain explicit Unsupported rather than unbounded callbacks. Other native
gate construction and full-engine arrays still need audit; final R1 acceptance
is not claimed. The next scoped slice controls column-based unitary/Kraus
validation, CP/TP checks for six native channel representations, direct Pauli
CSR roots and dense snapshot aggregates. Failed snapshot updates preserve the
previous cell. These allocation fixes are not additional SDK function parity.

The next R1 development slice also owns explicit named-gate/noise factories,
six native channel representations' conversions/algebra, dense evolution and
reset, integer-power products and analytic Aer ensemble density outputs inside
ResourceScope. Numeric stacked Kraus arrays are admitted without foreign
dispatch. Independent analytic phase/wire-order and surviving-view tests are
included. Tableau internals and unrelated public state/Operator methods remain
unclosed; a dense tableau preflight is not a complete allocation audit. Final
Task 3E validation includes three-environment FULL and the complete 28-section
local script, exit 0. Main full coverage is 90.1735% with 68 exclusions; seven
optional dependency subsets were not executed. This is line coverage, not SDK
function completion, real-cloud proof or a clean all-in-one dependency stack.
These changes do not complete the full Qiskit/PennyLane replacement goal.

C3-R2 state/Result development now adds native Pauli-eigenstate labels and
Statevector reset, plus scoped state/density sampling and defensive Result
copies with real buffer/view lifetimes. An explicit full-qargs reset preserves
the collapsed phase; the no-qargs default returns the canonical ground state.
Public optional Qiskit checks are numerical oracles, not native engines.
Object-level RNG, general subsystem dimensions and the complete public state
API remain incomplete. The initial slice's pending gates are superseded by the
final R2 receipts below. See the [state/Result progress report](docs/implementation/stage11p_c3_r2_state_result_progress.md).
Further P1 development adds native reflected Operator matrix arithmetic,
stable global-phase equivalence, admitted Pauli labels/reversal and shared-memo
Operator/Channel deep copies. It has focused mathematical and framework
evidence, not final R2 full/matrix acceptance. See the
[Operator/Channel progress report](docs/implementation/stage11p_c3_r2_operator_channel_progress.md).
P2 development now owns bool-tableau/GF(2) storage, signed Clifford composition/
adjoint/synthesis and Stabilizer measurement/sample outputs. Native 100-qubit
tableau execution remains non-dense while full matrix/state expansion rejects
before exponential allocation. A scoped direct Device density-output ownership
defect was also fixed using the existing density implementation. Old dense
allowances are retained pending proved replacement; universal core API parity
is not claimed. Final R2 same-tree FULL/framework/28-section matrix test gates
now finish at local code checkpoint b1d345b. Main and matrix full coverage both
record 90.35145452152166%, with 68 exclusions unchanged. Seven optional matrix
subsets were not executed and existing pip conflicts remain; no push, new-SHA
remote CI or real cloud hardware proof is implied. C3 stays PARTIAL after the
finite [joint review](docs/review/stage11p_c3_joint_review.md), with Stage 12-A
actual state API behavior next. See the
[Tableau progress report](docs/implementation/stage11p_c3_r2_tableau_progress.md).

Stage 12-A P0.1 now verifies bounded native mixed-radix state/density APIs:
explicit subsystem dimensions, full matrix evolution, tensor/copy/conversion,
probabilities, sampling, collapse/reset and partial trace. Fresh controlled
SDK-free/main/alternate FULL and the actual 28-section matrix pass; main and
independent matrix coverage are 90.44385395138274%, exclusions 68 unchanged.
Seven absent-dependency subsets were not executed; known environment conflicts
and the separate dirty-workspace Router attribution issue remain disclosed.
These are bounded native slices, not full state/public-API or SDK acceptance. Native
q0-first/dict probabilities/density shape-tuple conventions are preserved;
public compatibility, mixed mapped operators and general mixed-radix channel/
analysis APIs remain partial or unsupported. No push, remote CI or real cloud
execution is claimed. See the [P0.1 report](docs/implementation/stage12_a_p01_mixed_radix_progress.md).

Stage 12-A P0.2-A adds bounded native mixed-radix local numeric operators,
including reversed/noncontiguous qargs, Operator input/output dimensions,
rectangular density maps and full typed rectangular evolution. Fresh SDK-free,
main/alternate FULL, four-framework focuses and the real local matrix finish
with exits 0; seven missing optional subsets remain unexecuted and known pip
conflicts remain. Independent main/matrix line coverage is 90.4597%, not API
completion. Public calling conventions
and measured numerical differences remain explicit, and this is not complete
state/Operator/channel or SDK parity. See the
[P0.2-A report](docs/implementation/stage12_a_p02_mapped_operator_progress.md).

Stage 12-A P0.3 adds bounded native mixed-radix Operator composition, including
rectangular local maps and actual input/output-side propagation. Focused native
and explicit public-reference tests, fresh three-environment FULL/framework
gates and the complete local matrix finish with exits 0. Seven optional subsets
remain unexecuted, known pip conflicts remain, and independent main/matrix line
coverage is 90.4753%, not SDK function completion.
Native multiplication order is retained, not silently relabeled as Qiskit parity.
See the [P0.3 report](docs/implementation/stage12_a_p03_operator_composition_progress.md).

Stage 12-A P0.4 implements native mixed-radix local observable expectation for
Statevector and DensityMatrix, including reversed targets and complex
non-Hermitian results, without embedding a whole-system observable. Development
focuses pass 183 cases in each installed SDK environment and 171 native cases
without SDKs. Fresh frozen-runtime SDK-free/main/alternate FULL and the original
28-section matrix terminate with exits 0; main and matrix independently record
90.4959% coverage, 68 exclusions unchanged. Seven missing-dependency matrix
subsets remain unexecuted and known environment conflicts remain disclosed.
Public partition, scalar and target-validation differences remain explicit;
this is not complete state-class or Qiskit/PennyLane parity. See the
[P0.4 report](docs/implementation/stage12_a_p04_observable_expectation_progress.md).

Stage 12-A P0.5 implements native state arithmetic, Statevector trace/purity,
strict-partition inner products and density dictionary export. It corrects
complex density purity and tiny/rounded export values, and admits introduced
numeric outputs/conversions before allocation. Development focuses pass 288
cases in each installed environment and 274 native cases without SDKs. Fresh
frozen SDK-free/main/alternate FULL have 3302/4032/4030 passed, with skips/xfails
disclosed. Frameworks and the original 28-section matrix exit 0; main/matrix
coverage agrees at 90.6213%, exclusions 68 unchanged. Seven optional subsets
remain unexecuted and two known environment conflicts remain. Public coercion/exception, scalar
and configuration boundaries remain Partial; physical simulator admission keeps
its stricter tolerance. This is not full class or SDK parity. See the
[P0.5 report](docs/implementation/stage12_a_p05_state_arithmetic_progress.md).

Stage 12-A P0.7 adds actual native DensityMatrix.partial_transpose(qargs) for
mixed-radix subsystem row/column index exchange, retaining complex entries and
negative partial-transpose eigenvalues without normalization or PSD projection.
Focused tests pass 197 cases in each installed reference environment, or 149
native cases with 48 optional reference skips without SDKs; bounded broad
regression has 1501 passed. Fresh frozen SDK-free/main/alternate FULL have
3451/4229/4227 passed with skips/xfails disclosed; frameworks and original
28-section matrix exit 0. Main/matrix coverage agrees at 90.6279%, excluded 68.
Seven missing-dependency subsets and two known environment conflicts remain. Native
owned-copy and strict-target conventions retain explicit public compatibility
limits. Correct reverse/partial_trace are not rewritten. This is not complete
State/DensityMatrix or Qiskit/PennyLane parity. See the
[P0.7 report](docs/implementation/stage12_a_p07_partial_transpose_progress.md).

## Native Molecular Chemistry Checkpoint

`quantumbridge.qchem` now computes a bounded hydrogen/STO-3G s-shell workflow
from actual geometry: normalized Gaussian integrals, iterative closed-shell
RHF, spin-orbital fermionic Hamiltonian, Jordan-Wigner and number-sector exact
energy. H2 uses four untapered qubits. Native excitation circuits support
first-order shift/adjoint and optional Torch/Autograd VQE parameter gradients.

```bash
python -m examples.native_qchem_h2_integrals
python -m examples.native_qchem_h2_scf
python -m examples.native_qchem_h2_vqe
```

This is not the legacy fixed-coefficient H2/LiH demo, and is not complete
PennyLane qchem: other elements/bases, open-shell solvers, active spaces,
properties and fully differentiable molecular geometry/SCF are not implemented.
Supported native template/transform subsets now execute; their full contracts
and the complete chemistry library remain partial.
See the [chemistry checkpoint report](docs/implementation/stage11o_native_qchem_report.md).

## Native Serialization Checkpoint

`quantumbridge.serialization` provides native QASM2/QASM3 string/file APIs,
typed runtime-source execution, real public QPY v13 binary I/O and explicit
cross-format conversion. Existing Circuit/IR, compiler, native QNode gradients
and MPS execution are reused, not replaced by upstream SDK execution.

Default export rejects unsupported semantic loss. Optional external sidecars
preserve native layout/schedule/provenance metadata, but are not standard QPY
fields. QPY dynamic/custom/layout branches, non-dt delays, other QPY versions,
OpenPulse/calibration, general arrays/types and complete symbolic protocols
remain unsupported. QASM3 uses standard U phase semantics; importing Qiskit
2.4/2.6 source with its U3 convention requires explicit `dialect="qiskit"`.

See the [Stage 11M report](docs/implementation/stage11m_native_qasm_qpy_serialization_report.md),
[current parity ledger](docs/compat/core_parity_ledger.md), and
`examples/native_qasm*_roundtrip.py` / `examples/native_qpy_binary_roundtrip.py`.
Full Qiskit/PennyLane replacement and real hardware verification remain
incomplete. Studio remains deferred.

## Legal and Attribution

- Apache-2.0 plan: [docs/legal/apache2_compliance_plan.md](docs/legal/apache2_compliance_plan.md)
- Third-party source policy: [docs/legal/third_party_source_policy.md](docs/legal/third_party_source_policy.md)
- Qiskit ecosystem attribution: [docs/legal/qiskit_ecosystem_attribution_v0.1.md](docs/legal/qiskit_ecosystem_attribution_v0.1.md)
- PennyLane ecosystem attribution: [docs/legal/pennylane_ecosystem_attribution_v0.1.md](docs/legal/pennylane_ecosystem_attribution_v0.1.md)
- Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- Source migration ledger: [docs/migration/source_migration_ledger.md](docs/migration/source_migration_ledger.md)

Suggested public description:

> QuantumBridge is an independent quantum SDK that provides adapters and compatibility layers for common quantum computing workflows.
