# Third-party Notices

## Public MPS QNode Derivative Integration

`quantumbridge/diff/mps_derivatives.py` independently composes native local
matrix-function jets with the existing scaled MPS propagation kernel and
QNode/Device execution plans. Its Pauli contractions, variance product rule,
probability Walsh transform and resource/routing bounds are independently
authored mathematical implementations. No external simulator is called.
The local value path in `diff/matrix_power_jet.py` reuses native matrix
functions without whole-Gate/Operator materialization. Unbound transient
definitions are validated structurally rather than at a dummy numeric point.

The Torch bridge's native forward-AD/mixed rules use public custom-Function
APIs, with documentation as an API reference only:
https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.BackwardCFunction.html .
TensorFlow mixed-mode experiments exposed additional failures; those runtime
experiments were removed, with blocking tests retained. The public TensorFlow
autodiff documentation was consulted, not incorporated:
https://www.tensorflow.org/api_docs/python/tf/autodiff/ForwardAccumulator .

`scripts/probe_native_mps_power_oracles.py` independently constructs closed
SU(2) powers using the already installed mpmath at70 decimal digits. The JSON
fixture is numerical test data; mpmath differentiation is test-only, not a
production derivative algorithm. No SDK/library implementation or tests
were copied, no dependency installed and no cloud task submitted. Public
MPS integration is a development checkpoint, not complete SDK replacement
or completed four-framework mixed-mode acceptance.

The independent pure TensorFlow boundary/callback diagnostic scripts and
numeric JSON in `tests/interfaces/probes/` use public APIs and small
analytically specified test functions. They record wrong finite results and
exceptions, not a proven native quantum fix. Nested custom-gradient public
documentation was consulted as an API reference only:
https://www.tensorflow.org/api_docs/python/tf/custom_gradient .
No TensorFlow source or tests were incorporated. Failed diagnostic rules
were not inserted into the production graph adapter; the full replacement
objective and TF mixed-mode compatibility gap remain open.

The TF3 explicit VJP model and its host-callback diagnostic are independently
authored using elementary test functions and public TensorFlow gradient
APIs. Their analytic model succeeds but callback integration still fails;
neither result implies native SDK parity. The barrier/delay admission fix
only aligns native resource bounds with the existing noiseless simulator's
directive semantics. No TensorFlow/Qiskit/PennyLane implementation, tests,
package payload or runtime environment is incorporated by this increment.

The TF4 pure-tensor experiment in
`tests/interfaces/probes/probe_tf_native_tensor_execution.py` independently
expresses local rotations, coherent controlled operators and fixed-real
SU(2) principal powers using public TensorFlow tensor math. Its small-angle
coefficient recurrences and branch validation are mathematical derivations,
not copied TensorFlow or quantum SDK code. Installed TensorFlow source was
inspected to diagnose nested-AD errors; none was incorporated or patched.
The 70-digit numerical fixture is reproducible with the independently
authored `scripts/probe_tf_tensor_execution_oracles.py` and already installed
test-only mpmath. No production evaluator delegates derivatives to mpmath.
These tests are not a production QNode/tensor-network specialization and do
not close the existing100-wire TensorFlow mixed-mode failures.

The subsequent internal `interfaces/tensorflow_native_execution.py` local
kernel consumes existing native ExecutionPlan occurrences and the native
IR/parameter recorder. `interfaces/tensorflow_native_math.py` adapts our own
previously verified pure TF prototype formulas, not third-party code. It
uses public optional TF ops and existing native fixed-gate constants. The
independent no-QuantumBridge prototype remains separate as a test witness.
The kernel is not selected by public QNodes and is not a Device or MPS
executor. Local admission bounds are not a graph-wide TF allocator account.
New tests compare full matrices/complex jets, native Frechet mixed directions,
format-restored sources, actual framework AD and early refusal. No upstream
SDK/TF implementation, test, dependency payload or private JVP rule was copied.
PennyLane's public QNode contract was consulted while examining max_diff;
no max_diff policy or routing change is accepted from that consultation:
https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html .

TF6 subsequently connects that independently authored local kernel to the
public QNode's existing native ExecutionPlan for a bounded one-to-four-qubit
whole-circuit TF tensor specialization. Fixed matrices reuse native gate
semantics; trainable state, phase and Pauli contractions remain live TF ops.
The first VJP and streamed second rule use public TF custom-gradient and
ForwardAccumulator APIs, not an upstream quantum implementation, private
JVP registry, numeric quantum callback or injected complete Hessian.
Shared structural work admission is not runtime graph-wide accounting;
managed graph requests remain explicitly unsupported. This does not close
the100-wire MPS failures or establish complete QNode/max_diff parity.

`scripts/probe_tf_whole_qnode_oracles.py` independently generates complete
complex state and first/second derivatives at70 digits using test-only
mpmath, explicit basis-state wire maps and elementary SU(2) identities.
`scripts/probe_pl_tf_derivative_order.py` records black-box observations of
the two already installed PennyLane versions using a single-RY public API
example. Its JSON includes exceptions, disconnected and nonfinite results;
it is not production execution, copied SDK code, or a general parity claim.
No package source, tests, environment, credentials or hardware data were
incorporated. No dependency installation was performed for this increment.

The subsequent TF6-M2 eager-only managed increment reuses the existing
UnifiedResourceLedger reservation and cumulative-consumption operations.
It records conservative attempted tensor work and completed logical tensor
evaluations, not actual SDK/Device calls, CPU FLOPs, TF allocator memory or
retained TF result storage. Deferred native first/second rules keep their
original account. Managed cached graph requests remain unsupported; direct
independent ConcreteFunctions have static admission/runtime assertions only.
Concurrency and cancellation/error-lifetime tests are independently authored
using Python/TF public APIs. No third-party implementation is incorporated.

TF6-M2B subsequently adds an experimental, explicitly controlled cached
native TensorFlow handle and shared immutable derivative-request descriptions.
Explicit value/gradient/Jacobian/JVP/VJP/HVP/Hessian queries use our existing
native plan and resource ledger; public TF operations compute the values.
The compiler retains a mathematical snapshot and requires the original
owner's live execution/resource contract. Graph construction and controlled
synchronous calls record conservative logical work, not allocator/RSS or
arbitrary external graph costs. Direct internal ConcreteFunction calls remain
OUT_OF_SCOPE. Cached external reverse-over-forward correctness is NOT
accepted: two formal native tests remain red. M2 is therefore still PARTIAL;
the original four100-wire TF MPS failures and full replacement remain open.

`tests/interfaces/probes/probe_tf_cached_mixed_boundary.py` is independently
authored public-TF diagnostic mathematics, without any quantum SDK imports.
Its cubic reference uses elementary closed-form derivatives; the other two
diagnostic references are inline TF TapeTape cross-checks, not independent
mathematical oracles. The already installed TF2.21 runtimes exhibit mismatches
or disconnection at the cached reverse-over-forward boundary in this probe.
That observation is scoped to these installed runtimes, not a claim about
all TensorFlow releases. No upstream source/test, package payload, private
JVP patch, callback quantum calculation, credential or hardware result is
incorporated. Source port count remains zero; no dependency was installed.

The next TF6-M2C experiment adds the independently authored public-TF probe
`probe_tf_trace_inline_boundary.py`: elementary scalar/vector/complex models,
analytic values/Jacobians/Hessian contractions, two nonlinear losses and
three actual mixed-AD nestings. Cached-inner, outer-trace-inline and
eager-inline boundaries are compared without importing QuantumBridge or
quantum SDKs. Graph inspection uses public Graph/GraphDef APIs, not private
gradient registries. Its successful inline mathematics is diagnostic only;
there is no accepted native inline integration, per-external-graph ledger
guarantee, cached-RoF fix or M2 completion in that experiment.

TF6-M2D subsequently adds an explicit native trace-inline QNode subset.
FrozenParameterBinding reuses the existing immutable native ParameterExpression
syntax and compiler scalar evaluator; no TensorFlow graph is reverse parsed.
The existing local and whole-state TF kernels have an opt-in read-only binding
path. Default ordinary Gate/managed cached dispatch is unchanged. Native
construction happens once, then live TF scalars evaluate frozen occurrences
without rebuilding/copying Circuit operations or invoking a cached trainable
inner function. Existing boolean validation functions remain separate.

This mode is distinct from managed cached queries: native static bounds and
runtime numeric assertions are not per-external-graph Python ledger checks.
Owner/config checks apply when the Python inline entrance is invoked; direct
cached outer graphs do not recheck them. Eager mixed AD and AD traced together
with inline operations are validated subsets. Mixed AD outside an already
cached outer value graph is NOT_ACCEPTED, not generally intercepted. The two
old cached-callable RoF failures and four MPS failures remain open.

`scripts/probe_tf_inline_oracles.py` reuses our independent test-only70-digit
SU(2)/basis-map generator for a second point and complete complex-state jets.
No quantum SDK or TF is imported by that generator. No upstream implementation,
tests, private gradient registry, host quantum callback, complete-Hessian
injection, dependency payload, credential or hardware result is incorporated.
Source port count remains zero; full independent SDK replacement is incomplete.

TF6-M2E reuses the existing native method selector and immutable derivative
requests for read-only per-handle execution contracts, without a global
capability registry. Managed explicit queries, trace-inline mathematics and
experimental cached callable AD are distinct. The diagnostic callable has no
verified second-order contract; known external cached-RoF failures are not
deleted, relabeled or converted into a success claim. Declared unsupported
requests can be refused at controlled entrances, but opaque external AD cannot
be universally intercepted.

The first-order public-TF point/cotangent guard experiment is independently
authored elementary cubic mathematics, not upstream source. The native guard
separates cotangent transposition needed by genuine first-order forward AD
from an unapproved second quantum-point derivative. This is not a cached-RoF
repair. The performance probe measures the existing frozen native plans,
public GraphDef statistics and optional allocator observations. It does not
cache input-dependent tensor values or inject reference derivatives. Static
workspace bounds, unavailable observed workspace, Python object collection
and process RSS are explicitly different quantities. No new dependency,
private TF registration, third-party code, cloud data or financial evidence
is incorporated; source port count remains zero.

TF6-M2F corrects native explicit-query seed validation and mathematical-plan
integrity. A common validator rejects missing, empty and inapplicable request
seeds; the existing whole-plan class now fingerprints its actual scalar,
local-body, control, fixed-matrix and observable store. Managed provenance
also hashes its already-admitted classical mapping graph, without importing
or reverse-parsing a graph. It does not cache trainable numerical tensors.

Expanded default Jacobian tests exposed an existing eager/pfor integration
failure. Attempted graph/eager-callback adapters were withdrawn after real
second-order tests returned wrong finite values or vectorization errors.
No such adapter is retained in the default execution path. The independent
public-TF/NumPy sin/cos probe records these negative boundaries separately;
its diagnostic process exit is not a mathematical pass. No TensorFlow source
implementation or tests are copied. Reading official documentation and the
installed vectorizer for diagnosis does not constitute a source port.
Full native TensorFlow/SDK parity remains incomplete; source port count zero.

TF6-M2G subsequently repairs the bounded default eager-Jacobian/pfor path
using independently authored native whole-plan TensorFlow operations. It
reuses the existing ExecutionPlan, method selector, local statevector kernels
and actual native shift recipes, without another circuit store or a quantum
host callback in the traced VJP. Ordinary eager calls keep their existing
algorithms. The new quantum Jacobian is independent of its incoming cotangent
so nested derivatives retain quantum as well as classical loss curvature.
Only the explicitly tested analytic float64, fixed-observable, small static
statevector domain is covered; this is not general backprop support. Python
per-external-graph-execution accounting remains OUT_OF_SCOPE. Existing cached
RoF and 100-wire MPS failures remain separate, formal unresolved failures.
The additional scalar/sin-cos/complex pfor controls use public TF and NumPy
only. No upstream source or test port, private gradient registration, package
installation, vendor, cloud job or financial evidence change is involved.
The broader SDK-free regression also exposed a preexisting NumPy1.26 CPU
DLPack keyword incompatibility. The independently authored borrower now
selects the published pre2.1 protocol or modern copy=False contract without
retrying producer errors or introducing implicit array copies. Only public
NumPy API documentation was consulted; no NumPy source was ported.

TF6-M2H subsequently retains the actual native Circuit parameter-expression
source at managed construction, when its input is passed unchanged. It uses
the existing FrozenParameterBinding and readonly whole-plan tensor math for
the same controlled eager callable. Explicit managed queries still use real
cached functions; the native callable does not run an internal cached scalar
or quantum function during AD. Unsupported native scalar expressions preserve
their prior TF query route, without rerunning user Python; budget errors do
not trigger fallback. This is independently authored QuantumBridge code, not
an upstream source/test port or private TF gradient registration.

The original two cached-RoF tests pass on this native-source route. Six wider
TF-only/remapped/mixed-source RoF cases remain formal failures. External
cached-graph per-execution Python ownership remains OUT_OF_SCOPE, and the
100-wire MPS failures remain unresolved. M2I source probes only inspect our
captured scalar data and native operation lineage; no GraphDef interpreter,
GraphDef execution import or scalar-to-native lowerer is implemented by M2H.
Full native functional replacement remains incomplete; source port count zero.

TF6-M2I subsequently adds an independently authored, construction-only import
of a restricted scalar GraphDef into the existing native ParameterExpression
and FrozenParameterBinding. It is not a TensorFlow execution-graph importer or
general interpreter. It accepts only validated static float64 smooth scalar
operations and bounded layouts; int32/int64 data serves structural metadata,
not trainable arithmetic. Only the literal exponent2 square is admitted.
Public TF StridedSlice and ReverseV2 API descriptions and actual installed
public runtime behavior inform layout validation:
https://www.tensorflow.org/api_docs/python/tf/raw_ops/StridedSlice and
https://www.tensorflow.org/api_docs/python/tf/raw_ops/ReverseV2.
No upstream implementation, tests or documentation prose were copied.

The six broader scalar-capture RoF cases now pass unchanged in both local
environments. M2 remains PARTIAL, M3 NOT_STARTED, the100-wire MPS failures
remain unresolved, and full native functional replacement remains incomplete.
External cached-graph Python resource accounting is still OUT_OF_SCOPE.

TF6-M2J separates a known scalar-lowering capability miss from an identity,
domain, safety or resource refusal. Only an independently admitted unchanged
original public TF graph may preserve an old explicit query. Native AST
capture also retries only its two named frozen-scalar subset misses. Public
graph descriptors and fixed eager numeric Tensor data are validated without
executing GraphDef data or replaying user Python; closed constant hashes join
the snapshot identity. This does not add native scalar operators or extend
the M2I lowering whitelist. No TensorFlow implementation/test source, private
gradient registration or upstream quantum SDK execution engine is copied.

Further independently authored tests inject exceptions at constructor stages
and actual quantum invocation, check ownership/control/global-phase/UUID
snapshots and separate scalar construction work from runtime request work.
The development gate runner records source manifests before/after and actual
same-process import origins. These selected receipts are not FULL, production
certification, allocator/RSS observation or Git/staged/commit equality.
M2 remains PARTIAL, M3 NOT_STARTED, complete native replacement incomplete.

The M2J source-data checks subsequently use the public TensorFlow floating/
complex dtype classification rather than NumPy's storage-kind character.
This independently repairs nonfinite bfloat16 fixed-source data escaping the
legacy-query admission. It is a source-validation repair, not new native
bfloat16 scalar lowering or a claim of general mixed-precision support.

## Native Scaled MPS Tangent Kernel

`quantumbridge/information/mps_tangent.py` and
`quantumbridge/diff/mps_tangent.py` are independently authored scaled-vector
and mixed-product-rule kernels. They reuse the native MPS contractions,
SWAP routing, QR/SVD and managed array reservations. Complex scales and
coherent cross terms are preserved instead of treating derivative matrices
as unitary gates or probability mixtures. No upstream implementation, test
source, simulator or dependency is incorporated or installed.

The new tests compare bounded local Frechet matrices with the already
documented independent 70-digit fixtures, elementary rotation derivatives
and Python standard-library Decimal products. They execute 100-wire tensor
chains while forbidding dense state/Operator/Gate materialization. This is
an internal propagation-kernel checkpoint, not completed public MPS QNode,
framework, format-roundtrip acceptance or full SDK replacement. Existing
normal branch-normalization semantics are retained. Numerical-null Schmidt
removal is reported and is not an exact-arithmetic/error-certification claim.

## Native Mixed Power Directions And Forward AD

The native matrix-power implementation now also evaluates two different
Frechet directions and the actual mixed source derivative using a four-block
triangular matrix function. `quantumbridge/diff/matrix_power_jet.py` composes
native symbolic scalar, Gate-body and global-phase directional jets; the
existing adjoint engine streams second directions without constructing a
gate-parameter Hessian. `quantumbridge/interfaces/qnode_autodiff.py` registers
independently authored Autograd JVP and mixed-mode rules through Autograd's
public extension API. Its documentation is an API reference, not copied code:
https://github.com/HIPS/autograd/blob/master/docs/tutorial.md .

`scripts/probe_native_mixed_power_oracles.py` uses the already installed
mpmath 1.3.0 as an external test-only 70-digit oracle. Its explicit SU(2) and
four-dimensional matrices are independently authored. Checked-in JSON values
are numerical test data, not third-party implementation source. No mpmath,
Autograd or SDK source/binary is incorporated, and no dependency was installed.
General moving-exponent and nonunitary Frechet rules, new general MPS
derivatives and full SDK replacement remain incomplete.

## Native Moving-Projector Power Jets

`quantumbridge/information/matrix_functions.py` and
`quantumbridge/diff/matrix_power_jet.py` independently implement local first
and second matrix-power derivatives using triangular block matrix functions.
The existing native Schur/binomial matrix-function engine is reused; there
is no commuting-derivative shortcut, copied SDK source or upstream runtime
call. Nicholas Higham's public mathematical explanation of Frechet
derivatives is a reference:
https://nhigham.com/2020/06/23/what-is-a-frechet-derivative/ .
Installed Qiskit and the four autodiff frameworks are external test oracles
and integrations only. The Qiskit-free numerical example also exercises the
new native adjoint path. This increment covers bounded generic Gates with
one free scalar parameter and fixed exponents, not arbitrary multi-parameter
matrix functions, full device parity or full SDK replacement.

## Native Symbolic Spectral Power Continuation

`quantumbridge/diff/symbolic_power.py` independently lowers branch-consistent
fixed-projector powers to the existing native phase/Clifford/control engine.
The first and second derivatives are analytic classical traces composed with
the existing exact shift or adjoint rules, not numerical finite differences.
General moving-projector matrix families remain explicitly unsupported by
this lowering rule. Independent tests use scalar complex phases and Pauli
projectors; no upstream implementation or test source was copied. Existing
optional autodiff frameworks are exercised through their public interfaces,
not incorporated or installed by this increment. This is not full SDK parity.

## Offline Industrial And Security Application Studies

`quantumbridge/applications/industry.py` independently composes bounded native
cost models and QAOA circuits. Google job-shop definitions and Qiskit
Optimization modeling documentation are mathematical references, not copied
code. General diagonal truth tables are not asserted to be scalable QUBO
converters or production industrial systems.

`quantumbridge/applications/security.py` independently composes BB84, a
four-search-bit Grover example and simulated stream diagnostics. Grover's
paper, IBM BB84 teaching material and NIST entropy-source documentation are
conceptual references. Illustrative health thresholds, Hamming(7,4) and
Von Neumann processing are not a security certification or a QKD stack.
ML-KEM-768 and ML-DSA-65 use the installed OpenSSL command-line provider;
AES-GCM uses the installed `cryptography` package. These dependencies remain
external, with their own licenses; no source, binary, private key or wheel
is incorporated. NIST standards and provider documentation are linked in the
notebooks. Classical cryptography is not described as quantum hardware work.

## Offline Scientific Application Studies

`quantumbridge/applications/common.py` and `science.py`, the independently
authored tests under `tests/applications/`, and the application notebook
generator implement bounded native teaching workflows. No third-party SDK
source or tests were copied. Existing native chemistry, fermionic mapping,
state preparation, commuting-Pauli evolution and statevector APIs are reused.
Public PennyLane chemistry documentation, Qiskit Nature lattice definitions,
IBM time-evolution documentation and an HP lattice-model research paper are
conceptual references linked in the notebooks, not incorporated source.
The cases are local-only and do not establish full SDK or industry parity.

## Finance Research Case Studies

### Real-Stock CQM and Quantum Monte Carlo Extensions

The optional applications call installed D-Wave `dimod` (Apache-2.0), PennyLane
(Apache-2.0), SciPy and PyTorch public APIs. No SDK implementation or tests are
vendored. `ExactCQMSolver` performs classical local enumeration; PennyLane's
matrix-based QuantumMonteCarlo reference is simulator-only. Native gate-model
QAOA and maximum-likelihood amplitude estimation are separately identified.
No D-Wave cloud or Quafu hardware submission is performed by these applications.

`quantumbridge/finance/data/plotly_stockdata.csv` is a real historical stock-data
snapshot from Plotly's public datasets repository, pinned at commit
`0f7c47868a3393316098a8bf97d548e1ee23d22a`. Its source URL, exact SHA-256, dates,
limitations and repository-published MIT license are retained in adjacent
`provenance.json` and `PLOTLY_LICENSE.txt`. Copyright (c) 2019-2024 Plotly
Technologies Inc. The original upstream price-adjustment methodology is not
documented; no audited total-return or current-quote claim is made. These
historical research demonstrations are not trading recommendations, verified
production risk models, hardware acceptance, or evidence of quantum advantage.

`quantumbridge/finance/` independently implements five small synthetic finance
workflows with the project's native circuits, simulator and existing Quafu
transport. Public Qiskit Finance tutorials are application/mathematical
references only; no tutorial code or upstream implementation is copied.
The notebooks link original portfolio, credit-risk, option-pricing and
QuarkStudio interface documentation. Direct hardware sampling is explicitly
distinguished from QAE, from local classical training and from model error.
No quantum advantage, production finance, official endorsement or full SDK
parity is claimed. QuarkStudio remains an optional external dependency; its
source and credentials are not redistributed.

The finance hardware-acceptance continuation adds independently implemented
Wilson/DKW/empirical-bootstrap diagnostics and fixed-decoder comparisons using
actual authenticated Quafu task receipts. Raw counts and exact submitted QASM
remain separate from simulation. Existing public QuarkStudio APIs are called
as an optional dependency, not copied; no model accuracy, hardware acceptance
or full SDK parity is inferred merely from successful cloud execution.

The D1 continuation independently parses the service's returned public QASM2
and bounded U/CZ/Measure QLISP literal descriptions into native circuits. These
are execution receipts, not copied third-party implementation. Four fixed
zero/one preparation/measurement probes and Wilson-score diagnostics do not
constitute readout calibration, error mitigation, financial certification or
full Qiskit/PennyLane parity. Original finance evidence remains unchanged.

D2 independently computes finite count-order hypotheses, layout/time comparisons,
joint-error bootstrap and conditional basis-pattern identifiability from the
existing thirteen task receipts. These are offline mathematical diagnostics,
not copied provider algorithms, a readout model or validated error mitigation.
No new hardware tasks, SDK dependencies or upstream source are introduced.

D3 uses installed package metadata, public signatures/help and documentation
as contract evidence, not upstream implementation/test source. Independently
implemented label traces and read-only gateway guards neither rewrite provider
results nor infer undocumented raw-count serialization. Separate quafu-sdk
documentation is not treated as historical QuarkStudio response metadata.

## Native Gate Definitions Work In Progress

The R2M controlled-power continuation independently uses Pauli eigenprojector
identities and scalar complex branches for mathematical tests. Numeric Gate
powers reuse the existing native Operator engine; actual public Qiskit calls
are development observations only. Public inverse object classes and the
observed open-controlled Phase power defect do not supply implementation or
test source. No new dependency, vendored source or upstream runtime is added.
General symbolic fractional-power QNode compilation and full SDK parity are
not claimed.

The R2L fractional Operator continuation independently certifies exact binary
2x2 repeated spectra using standard-library scalar rational arithmetic, reuses
the existing native matrix function on exact disconnected components, and
screens branch-sensitive Schur pairs. Higham's published mathematical
identities and actual public Qiskit calls are references, not copied SDK/source
or tests. An already installed mpmath is a development oracle only, not a new
runtime dependency or matrix backend. Full SDK parity is not claimed.

P4-U3D develops independent native bit/register identity, structural native IR,
zero-width scalar operations and existing-engine integration. Actual installed
public Qiskit2.4.1/2.6.0 calls and IBM's public QPY protocol documentation are
behavioral references, not implementation/test source. No source is copied,
ported or vendored, and native calculations do not call an upstream executor.
U3D2 adds native public contexts/handles, measured standard invocation-frame
semantics, standard QPYv13 register records and a supported QASM register-array
subset. It reuses native Operator/Compiler/Devices/QNode/AD, not an upstream
executor. Complete public ABI and interchange parity are not claimed. The
published QPY register protocol and actual public calls, not implementation or
test source, inform this independent work. No credentials, cloud/hardware,
package installation or release operation is performed in this work.

U3D2-R continues independent native source-owner/cached-frame/context and
public factory work in the same engine. Actual public2.4.1/2.6.0 calls expose
different generic versus standard context and copy behavior, including matrix
UnitaryGate contexts with no scalar params. Native scalar matrices, coherent
control, IR and bilateral QPY use native validation/math and published binary
records, not an upstream executor. Independently authored tests and
`examples/native_public_circuit_frames.py` are recorded in the source ledger.
No upstream implementation or test source is inspected, copied or vendored.
Upstream stale generic owner/definition behavior is explicitly not emulated;
full ABI and functional replacement remain incomplete.

U3D2-R2 adds independent public binding/query and internal definition-binding
separation, using actual installed public method calls/signatures only as
behavioral observations. Native parameter UUIDs/formal lineage, graph admission,
atomic replacement and existing IR/QPY/AD implementations remain authoritative.
No upstream source is copied or delegated to, and explicit unsupported or
intentional safety differences are not counted as functional parity.

U3D2-R2 conversion work uses independent local bit/register normalization and
native public error contracts. An observed public2.4.1 Instruction-phase bug
and its actual2.6.0 fix are explicitly distinguished. The supported structural
QPY controlled-rotation export is derived from a commuting same-axis five-step
identity and independently tested against full analytic complex matrices and
real public loaders. It does not use a dense or upstream serialization engine.
Native bit, parameter, Gate/Instruction, QPY and AD implementations are reused;
metadata and remaining full-format/public API gaps remain explicit.

The R2 metadata-domain continuation independently separates public Python user
data, native executable state and standard portable metadata. Actual public
2.4.1/2.6.0 metadata calls are behavioral oracles; no upstream implementation
or tests are inspected or copied. JSON standard QPY uses real phase/register
fields, and native IR uses typed execution records, not hidden user keys.
Opaque Python data is not JSON-copied in memory; unsupported transport and
managed copy callbacks are explicitly refused. Legacy fluent configuration
and one authoritative native operation store are retained. This is development
work, not full ABI/SDK acceptance or a claim of complete replacement.

The R2 owner-lifecycle continuation independently extends executable invocation
frames to generic ControlledGate and AnnotatedOperation, with bounded owner
admission, owned inverse bases and native IR name records. Actual public
Qiskit2.4.1/2.6.0 construction, mutation, Operator and QPY calls are behavioral
references, never native runtime delegates. Real QPYv13 custom Instruction and
annotation classical fields are encoded in published binary headers/operands,
not a private metadata sidecar. Deferred nonunitary annotation transport does
not grant unitary executability. Independent complex-matrix, four-interface
derivative, no-dense MPS and native-device tests exercise the existing engines.
No upstream implementation/test source is read, copied or vendored, and no
package installation, credential, cloud/hardware or release action is added.

The R2 controlled-base continuation independently implements bounded native QPY
multi-control parity decomposition and variable-width MCPhaseGate/MCXGate import
into generic native ControlledGate objects. Actual installed public API/QPY calls
are behavioral references, not copied SDK code or native runtime delegates.
Analytic projectors, control-block matrices and Pauli generators supply separate
mathematical oracles. Native first/second AD reuses existing execution engines;
controlled rotations account for the inactive-subspace zero eigenvalue. Native
IR explicitly retains invalidated formal history without treating it as active.
Measured public QPY/parameter-index and non-atomic error defects remain explicit
differences, not functional parity claims. No source vendoring, credential use,
package installation, hardware/cloud operation or Studio/Router change is added.

The R2 public ParameterView continuation is independently implemented in native
Python over the existing circuit parameter graph. Published IBM QuantumCircuit
parameter documentation and actual installed public2.4.1/2.6.0 snapshot, copy,
query and set-operator calls are behavioral references only. No upstream
implementation/test source is inspected or copied. Snapshot mutation is local;
native formal plans remain immutable tuples, and native IR/QPY, AD and execution
use their existing engines. Logical size/work admission and managed callback
refusal reuse existing native limits. No vendor, installation, credential use,
hardware/cloud call or Studio/Router change is added.

The R2J ParameterVector continuation independently implements native list
ownership, immutable names, bounded atomic identity construction and managed
scalar/vector admission. Actual installed public2.4.1/2.6.0 equality, vector,
resize, binding and standard QPY calls are behavioral references only. No SDK
implementation/test source is read or copied. The native list remains one
coherent sequence where public2.6 cache mutation can diverge; that difference is
explicit, not a MATCH claim. Empty vector names use standard QPY vector fields,
without hidden metadata. Existing AST, UUID/formal lineage, IR and AD engines
are reused. No package installation, vendor, credential or cloud/hardware use.

P4-U3C independently implements native composition, identifier scope/remapping,
checked operation-owner reuse and decoded-symbol interning. Actual installed
public Qiskit2.4.1/2.6.0 calls and published IBM API/protocol documentation are
behavioral references only. No upstream implementation or test source is read,
ported or vendored. Native ControlBlock, simulators, compiler, IR/QPY and AD
are reused; the native executor never delegates these computations upstream.
Unsupported timing/transport/public ABI remains explicit. No new package or
environment installation, authentication, cloud/hardware or token call occurs.

P4-U3B independently implements native Instruction construction/inverse/repeat,
owned invocation references, Circuit return/name contracts, native IR and
published QPY v13 custom Instruction records in the existing engine. Actual
installed public Qiskit2.4.1/2.6.0 calls are behavioral/binary interoperability
references only. No upstream implementation or test source is read or ported.
Definition/resource/storage, lowering, control-flow, simulators and AD remain
native QuantumBridge mechanisms. Explicitly unsupported transport/execution
contracts are not replaced with upstream calls or mislabeled as full parity.
No third-party source, package/environment installation, vendor artifact,
cloud/token/hardware call or new quantum engine is introduced.

Stage 12-B P4 independently implements native unitary definition graphs,
structural transformations and controlled phase in the existing architecture.
Behavioral references are IBM's public Gate API and installed Qiskit public
calls, not upstream implementation/test source. Numerical spectral powers use
the existing NumPy linear algebra runtime; no external quantum engine or
source port is introduced. Later P4-S independently implements native custom
definition IR and supported standard QASM2/3 and QPY v13 records. Published
IBM QPY and OpenQASM language specifications, plus public API black-box calls,
are references, not implementation source ports. Unsupported records and
observed upstream source-export losses remain explicit. Full protocol and P4
acceptance are unfinished.

P4-T independently adds native ordered operation modifiers, deferred native
resolution and standard definition access using existing QuantumBridge
compiler recipes. IBM's public AnnotatedOperation/Gate API and published QPY
format documentation inform behavior and explicit unsupported boundaries;
installed Qiskit 2.4.1/2.6.0 public calls are optional black-box comparisons.
No upstream source/tests, new quantum runtime or vendor artifacts are added.
Native execution and SDK-free examples do not import the upstream SDKs.
P4-U1 independently adds supported QPY v13 `a` annotation / `m` modifier
binary records in the existing codec. Published IBM record layouts and real
public dump/load calls are the protocol/behavior references, not copied
implementation or test source. Unsupported symbolic modifier powers reflect
v13's float64 field; public v13 circuit parameter-index loss is separately
reproduced on both public releases and is not presented as repaired.

P4-U2 independently implements checked parameter views, atomic native formal
rebinding, validation and typed native IR binding records in the existing
core. Public Gate/Instruction/AnnotatedOperation API documentation and actual
public 2.4.1/2.6.0 calls are behavior references, not source ports. Native
lineage records are explicitly QuantumBridge IR, not invented standard QPY
fields. The existing native storage copier also initializes audited symbol
hash fields before reconstructing owned hashed collections; resource preflight,
foreign-protocol rejection and copy ownership remain in force. No upstream
source/tests, new quantum engine, environment or vendor payload is included.

P4-U3 independently implements real native standard Gate classes, safe owned
frozen/mutable objects and standard return/error contracts. Existing native
compiler recipes, parameter ASTs, quantum execution and differentiation remain
the implementation. IBM public APIs and actual installed public2.4.1/2.6.0
calls are black-box references only, never upstream implementation/test ports.
Published QPY v13 class/NumPy parameter records and NumPy's documented .npy
header APIs inform bounded matrix interchange. Native class identity is explicit
in QuantumBridge IR, never fabricated from a custom Gate's display name.
Public/private singleton identity and remaining ABI differences stay explicit.
No upstream source, SDK runtime wrapper, package installation, vendor payload,
cloud/token call or new execution engine is introduced.

Status: Stage 11P-C3 work in progress; C2A committed
Date: 2026-10-21

QuantumBridge is an independent project. References to Qiskit and PennyLane are for attribution, compatibility, and optional adapter documentation only and do not imply endorsement.

## Native Symbolic Mathematics

Stage 12-B P3 independently implements scalar expression AST functions,
symbolic differentiation, substitution and limits. Public IBM API/protocol
documentation and installed Qiskit public calls are behavioral oracles only;
no upstream implementation or tests are copied. The native calculus does not
import Qiskit, SymPy or SymEngine. Optional SymPy interoperability exports
native expressions outward and cannot preserve parameter UUID identities.
Optional framework math operations retain their own classical graph plumbing.
No additional vendored source or production symbolic engine is introduced.
The full SDK replacement goal remains incomplete; focuses do not prove parity.

## Native Template Verification

Stage 11P-C3-R1 Task 3E additionally applies independently authored allocation
contracts to existing native gates, noise, channel conversions/algebra and
dense evolution. Positive tests use mathematical identities and optional
installed public behavioral oracles, not ported implementation/test source.
No additional third-party code, quantum engine or vendored dependency is
included; this is not a claim of complete SDK parity or process RSS control.

Stage 11P-C3-R1 independently adds native array preallocation and storage-root
lifetime contracts around the existing plans. Optional Torch and Autograd
public differentiation APIs supply classical graph plumbing only. NumPy's
documented no-copy DLPack interface informs CPU input borrowing; no upstream
implementation/test source is ported. Independently authored weak-reference,
budget, matrix and framework tests are not full SDK/allocator certification.
Source ports remain zero; the full replacement objective remains incomplete.

The R1 persistent-storage slice independently adds construction-pool leases,
request-local resident admission and compiled-query ownership tests. Optional
TensorFlow's public numpy_function, Graph and RegisterGradient APIs inform
callback integration; their implementation/test source is not ported. The
registered status-gradient rejection has no per-request graph/kernel payload.
This is bounded native storage accounting, not complete framework/RSS parity.

The ordinary TF ownership slice also uses public function/IdentityN gradient
overrides, not private registry changes or upstream source. Strided tree copies
use documented NumPy destination copying and view/reshape behavior. Foreign
copy-protocol tests and all implementation remain independently authored.

The R1 MPS foundation independently adds allocation/ownership boundaries around
the existing native contractions and NumPy QR/SVD calls. Public NumPy API
shapes are references, not implementation/test source ports. Physics, noise,
SWAP and framework regressions use independently authored fixtures and public
black-box comparisons. Native snapshot/root transfers and explicit completed-
trajectory accounting are independently authored extensions of the existing
Context/Result/Job/ledger, not source ports or a second simulator. Hidden
allocator workspace and general serialization paths are not claimed fully
controlled; full replacement remains incomplete.

The observable/IR/JSON allocation slice independently implements dense Pauli
matrix filling from signed basis permutations, controlled numeric conversion
and explicit Hermitian-validation scratch. Existing public formats use the
Python standard JSON encoder/decoder with structural limits; no second
serializer, upstream implementation/test port or SDK quantum engine is added.
Python/NumPy internal allocator memory remains explicitly distinct from owned
numerical arrays and logical serialized byte estimates.

The native validation/CSR/snapshot slice independently applies column inner
products, existing Choi/Pauli-basis conversions and signed Pauli permutations
around existing native numerical routines. NumPy/SciPy remain classical array
dependencies, not quantum SDK engines. No implementation/test source ports or
vendored dependencies are added; full functional replacement is incomplete.

Stage 11P-C3 currently adds independently authored native Device requests,
resource reservations and structured derivative outcomes around the existing
native plans, plus explicit optional CPU JAX/TF derivative-status queries.
Frameworks supply classical source maps only; no quantum SDK implementation or
test source is read/copied as the native engine. This work is in progress,
not complete C3 or full Device/SDK parity. Focused observations do not establish
strict process memory bounds, complete framework graph budgeting or remote CI.

Stage 11P-C2A independently differentiates QuantumBridge's existing reverse-state
gradient recurrence in a tangent direction. Native gate matrices, observable
actions and statevector execution remain the engine. Jones and Gacon's public
adjoint-gradient paper is a mathematical reference, not implementation source
or a claim that its first-order algorithm supplies this second-order extension.
Tests use independent trigonometric/matrix formulas, native parameter shifts and
real classical framework graphs. No upstream implementation/test ports, private
quantum derivative helper, cloud/token/hardware call or vendored tree is added.
The complete functional replacement goal remains incomplete.

Stage 11P-C2 independently streams framework HVPs through the existing native
second-order engine. Optional Autograd public primitive VJP registration and
JAX documented extension namespaces supply classical transformation plumbing,
including explicit numerical transposition and CPU host lowering; no upstream
quantum derivative engine or implementation/test source is used. Public JAX
custom-derivative, callback and primitive documentation informs the contracts.
Finite-shot linearity, native MPS and one-build H2 examples are independently
authored. No source ports, vendoring, credentials or cloud calls are included.
Full independent functional parity remains incomplete.

Stage 11P-C independently implements convolved native shift recipes, second
classical tracing, transformed processor chain rules and bounded CPU framework
second derivatives. Existing native plans/devices/Pauli rules are reused;
Torch/Autograd/JAX/TF provide classical graph APIs only. Published interface
contracts are references, not copied implementation or tests. Public installed
PennyLane Hessians are optional black-box oracles; unresolved public ArrayBox
results are reported separately, never counted as numerical matching. Source
ports remain zero. No credentials, cloud/hardware calls, third-party source,
package artifacts or environments are included. Complete parity remains open.

Stage 11P-B independently connects existing native quantum execution and
shift/adjoint/transform Jacobians to optional TensorFlow classical graphs.
Public custom_gradient/numpy_function contracts inform the interface boundary;
no framework or SDK implementation/test source is copied. Native host-plan
machinery is shared with the previously independent JAX bridge, not a second
quantum engine. Public installed PennyLane calls are independent black-box
oracles only. No TensorFlow Quantum/PennyLane quantum runtime is used by the
native path. Source ports remain zero. No credentials, cloud/hardware calls,
upstream sources, environments, wheels or installed package trees are included.
This bounded CPU first-order checkpoint is not full SDK parity or endorsement.

Stage 11P-A independently connects the existing native QNode, Device,
shift/adjoint and transformed postprocessing Jacobians to optional JAX graphs.
JAX provides classical differentiation and public callback/transform APIs;
it is not a replacement quantum simulator. Public documentation supplies
interface contracts, not implementation source. Independently authored public
PennyLane 0.44.1/0.42.3 calls are black-box numerical oracles only, with known
unsupported public paths reported separately. No JAX/SDK implementation or
test source is ported; no private upstream gradient helper is used by native
execution. Source ports remain zero. Environments, wheels, installed package
trees, credentials and cloud/hardware calls are not included. This is not
complete JAX/PennyLane/Qiskit parity or an endorsement.

Stage 11O-B6 independently implements coherent measurement recording,
bounded controlled-unitary lowering and logical terminal reconstruction.
It reuses QuantumBridge's dynamic execution, compiler algebra, QNode and
gradient engines. Public PennyLane 0.44.1/0.42.3 calls are optional black-box
oracles; published documentation is a behavioral reference, not implementation
source. No SDK implementation/test code is ported or used as the native
execution engine. Source port count is zero. No credentials, environments,
package artifacts or upstream source trees are included. Supported subsets,
differences and remaining complete API contracts are documented explicitly;
this is not full SDK parity, production certification or endorsement.

Stage 11O-B5 independently implements first-order native transformed child
quantum Jacobians, bounded classical postprocessing differentiation and
gate/coefficient chain rules. Existing QuantumBridge shift/adjoint, Tape/QNode,
Device and interface engines are reused. Optional Torch/Autograd supply
classical graphs and optimizers, not quantum differentiation. PennyLane public
gradient APIs are black-box test references only; no SDK implementation/test
files or private SDK gradient helpers are opened/copied or used at runtime.
No third-party code, credentials, environments or package artifacts are
vendored. MPS resource receipts come from actual measurement calls. This is
not complete transform/gradient/device parity or upstream endorsement.

Stage 11O-B4 independently implements bounded native Pauli QWC grouping,
deterministic integer shot allocation, one actual Device.sample buffer per
finite-shot group, covariance statistics and source result reconstruction.
Existing QuantumBridge Tape/QNode/Device engines are reused. PennyLane public
documentation and 0.44.1/0.42.3 public calls are optional black-box references;
no SDK implementation/test files were opened or copied, no private SDK helpers
are imported, and no upstream source, package artifacts, credentials or
environments are vendored. Documented API/version differences remain; complete
transform/gradient parity and upstream endorsement are not claimed.

Stage 11O-B2/B3 independently implements native QNode/callable transform
dispatch, parameter reconstruction, inverse cancellation, bounded native
definition/basis decomposition, target compilation and single-term measurement
reconstruction. Installed PennyLane 0.44.1/0.42.3 is an optional public API
oracle, not the native execution engine. No upstream implementation/test files
were opened, copied or vendored. Full transform/API/gradient parity is not
claimed. Public complex-matrix comparisons explicitly retain GlobalPhase.

Stage 11O-B1 independently implements bounded native Tape batches, reverse
result reconstruction, safe rotation merging and a Hamiltonian term-Tapes
prototype. It reuses QuantumBridge's existing native QNode/Device execution
and compiler algebra; it does not copy or invoke an upstream transform
implementation. This is not complete PennyLane transform parity.

Stage 11O-A independently enhances existing native QML embeddings/layers and
adds concrete IQP, QFT and bounded Pauli product-formula templates. Public
PennyLane APIs are optional black-box behavior oracles only. Existing native
mathematics and execution engines are reused, not SDK template source or
tests. Standard gate serialization and explicit unsupported preparation
exports are tested. Full template/transform parity is not claimed.

## Native Molecular Electronic Structure

Stage 11O-C independently implements bounded s-Gaussian integrals, closed-shell
RHF, molecular fermionic/Jordan-Wigner construction and excitation circuits.
Only six hydrogen STO-3G numerical basis parameters from MolSSI BSE version 1
are included, with provenance and a conservatively retained project license
notice in `LICENSES/BSE_DATA_NOTICE.txt`. No BSE software is included. Public
PennyLane/PySCF calls are optional black-box oracles, not native solvers.
Geometry-to-Hamiltonian computation is native; complete qchem parity is not
claimed. See the Stage 11O chemistry checkpoint report for restrictions.

## Native Clifford And Stabilizer Verification

Stage 11N independently implements intermediate Save/Set execution, bounded
snapshot aggregation, CPU method selection and gate-attached CPTP/readout
noise. Public Aer documentation and installed public SDK calls are behavioral
references only. Native execution uses existing QuantumBridge state, tensor,
tableau, channel, Backend/Job and control-flow implementations. No upstream
source/tests, package artifacts or environments were copied or vendored.
Full Aer API/performance parity is not claimed; see the Stage 11N report.

Stage 11K independently implements classical first-order tracing, gate-occurrence
parameter-shift, inverse-state/reverse-cotangent adjoint, Jacobian products and
shot partitions. Optional Torch/Autograd supply classical graphs/optimizers;
QuantumBridge supplies quantum execution and derivatives. Public PennyLane APIs
are optional black-box test oracles only. No third-party implementation, tests,
package artifacts or environments were copied or vendored.

Stage 11J independently implements typed classical expressions/storage,
structured dynamic control flow and a native interpreter over vector, density,
MPS and supported tableau states. Optional public Qiskit/Aer and PennyLane APIs
are black-box test oracles only. No upstream source, tests or documentation
wording was copied into the implementation. Public references and remaining
semantic differences are recorded in the Stage 11J implementation report.

Stage 11I independently implements QR/SVD matrix-product-state tensors, local
gate contraction and nonlocal swap networks, native CPTP ensembles/trajectories,
compressed result serialization and device/backend/QNode integration. Installed
Aer MPS and PennyLane default.qubit are optional behavioral test oracles only.
Public mathematical/API references are recorded in the Stage 11I report.
No upstream implementation source, tests or package artifacts were copied.

Stage 11H independently implements local density-matrix contractions, physical
Kraus instructions, mixed-state execution and QNode integration. Installed
Qiskit/Aer and PennyLane default.mixed are optional behavior oracles only.
Phase-damping formulas follow the public mathematical channel definition.
No upstream source, tests or package artifacts were copied or vendored.

Stage 11G adds independent tableau execution, dynamic measurement/reset and
if/else handling, native noise trajectories and Backend/Job/primitive
integration. Installed Qiskit 2.4.1 and Aer 0.17.2 serve as optional behavioral
oracles; they are not required by the native execution path. No upstream
source, tests or package artifacts were copied or vendored.

Stage 11F independently implements signed binary symplectic Clifford algebra,
circuit synthesis, random Clifford sampling and pure stabilizer-state
execution. The mathematical basis is the published stabilizer formalism;
Qiskit 2.6.0 is an optional behavioral test reference, not a runtime dependency
for these implementations. No upstream source, tests or package artifacts
were copied or vendored. Mathematical and API references are listed in the
Stage 11F implementation report.

## Native Pauli Algebra Verification

Stage 11E independently implements phase-aware Pauli and Pauli-list algebra.
The optional verification environment uses Qiskit 2.6.0 and PennyLane 0.42.3
as behavioral references. Native execution requires NumPy; optional sparse
matrix output uses an installed SciPy. No upstream source, tests, or package
artifacts are copied into this repository.

## Qiskit attribution

Qiskit is an open-source quantum computing SDK distributed under Apache License 2.0.

- Local license copy: `LICENSES/QISKIT_LICENSE.txt`
- Current local package used for adapter tests: `qiskit==2.4.1`
- Current QuantumBridge usage mode: Adapter Integration and Upstream Dependency only.
- Source port status: No Qiskit source files have been copied into QuantumBridge in the current Stage 4 implementation.

## PennyLane attribution

PennyLane is an open-source quantum programming framework distributed under Apache License 2.0.

- Local license copy: `LICENSES/PENNYLANE_LICENSE.txt`
- Current local package used for adapter tests: `PennyLane==0.44.1`
- Current QuantumBridge usage mode: Adapter Integration and Upstream Dependency only.
- Source port status: No PennyLane source files have been copied into QuantumBridge in the current Stage 4 implementation.

## Apache License 2.0

The Apache License 2.0 text is stored at:

- `LICENSES/Apache-2.0.txt`

## Migrated source record table

| QuantumBridge module | Upstream project | Original path | License | Mode | Modification notes |
| --- | --- | --- | --- | --- | --- |
| None in current Stage 4 implementation | N/A | N/A | N/A | N/A | No source port performed |

Future source ports must add a row here and update `docs/migration/source_migration_ledger.md`.

## Qiskit ecosystem optional packages

QuantumBridge Stage 7 may optionally depend on Qiskit ecosystem packages for inventory, passthrough, adapter integration, and result wrapping. These packages remain upstream projects and are not vendored into this repository.

Packages covered by Stage 7 planning include:

- Qiskit core;
- Qiskit Aer;
- Qiskit IBM Runtime;
- Qiskit Finance;
- Qiskit Optimization;
- Qiskit Machine Learning;
- Qiskit Experiments;
- Qiskit Addons.

Usage mode: optional dependency, public API inventory, passthrough scaffold, and selected adapter integration. Source port status: no Stage 7 Qiskit ecosystem source files were copied into QuantumBridge.

## PennyLane ecosystem optional packages

QuantumBridge Stage 7 may optionally depend on PennyLane and PennyLane ecosystem plugins for inventory, passthrough, adapter integration, and result wrapping. These packages remain upstream projects and are not vendored into this repository.

Usage mode: optional dependency, public API inventory, passthrough scaffold, and selected adapter integration. Source port status: no Stage 7 PennyLane ecosystem source files were copied into QuantumBridge.

## Stage 6.1 optional upstream integrations

P2 planning and implementation introduces optional dependency routes for:

- Qiskit Nature
- Qiskit Algorithms
- PySCF
- OpenFermion

These packages are not vendored into QuantumBridge. QuantumBridge uses them only
when installed through optional extras and records behavior as Adapter
Integration or Upstream Passthrough unless a separate source-port review
explicitly approves otherwise.

QuantumBridge does not claim that upstream functionality is QuantumBridge
original work. QuantumBridge is not endorsed by IBM, Qiskit, Xanadu, PennyLane,
PySCF, or OpenFermion maintainers.

## Stage 7.2 additional optional ecosystems

Stage 7.2 adds optional inventory, passthrough, and result-schema adapters for Qiskit Nature, Qiskit Algorithms, Qiskit Dynamics, Qiskit Metal, PySCF, and OpenFermion, and expands the existing Finance, Optimization, Machine Learning, Experiments, Aer, and PennyLane coverage.

These upstream packages are not distributed inside QuantumBridge. Each package is installed into an isolated optional environment for verification. Qiskit Metal coverage is advisory and does not imply chip fabrication or electromagnetic simulation capability.

## Stage 8D Qiskit ecosystem adapter contracts

Stage 8D hardens optional adapter contracts for:

- Qiskit core;
- Qiskit Aer;
- Qiskit Nature;
- Qiskit Algorithms;
- Qiskit Finance;
- Qiskit Optimization;
- Qiskit Machine Learning;
- Qiskit Dynamics;
- Qiskit Experiments;
- Qiskit Metal;
- Qiskit IBM Runtime;
- Qiskit Addons.

Usage mode: optional dependency discovery, runtime public API inventory,
upstream passthrough where installed, QuantumBridge result-schema wrapping,
warnings, provenance, unsupported metadata, and environment validation.

Source port status: no Stage 8D Qiskit ecosystem source files, tests,
documentation prose, comments, error strings, wheels, dist-info, egg-info,
site-packages trees, or virtual environments were copied into QuantumBridge.

Runtime support is offline-only. QuantumBridge does not access IBM Cloud, read
tokens, store credentials, or submit jobs. Qiskit Metal support is advisory
only and does not provide chip fabrication, external electromagnetic
simulation, or layout signoff capability.

## Stage 9G Mitiq optional ecosystem

Stage 9G adds optional Mitiq dependency detection and independently
implemented educational native error-mitigation workflows. Mitiq remains an
upstream project and is not vendored into this repository.

Usage mode: optional dependency detection, upstream passthrough metadata,
QuantumBridge result-schema wrapping, warnings, provenance, and independently
implemented native educational ZNE/readout workflows.

Source port status: no Mitiq source files, tests, documentation prose, comments,
error strings, wheels, dist-info, egg-info, site-packages trees, or virtual
environments were copied into QuantumBridge.

## Stage 9H PennyLane-Qiskit optional ecosystem

Stage 9H adds optional PennyLane-Qiskit plugin dependency detection and
independently implemented educational native bidirectional bridge workflows.
PennyLane-Qiskit remains an optional upstream package and is not required by
the native Stage 9H examples.

Implemented by QuantumBridge:

- Qiskit circuit to QuantumBridge IR to PennyLane executable spec;
- PennyLane operation/tape/QNode metadata to QuantumBridge IR to Qiskit circuit;
- bidirectional Bell-state equivalence proof over QuantumBridge's native simulator;
- result schemas, warnings, provenance, and no-cloud/no-token metadata.

Source port status: no PennyLane-Qiskit, Qiskit, PennyLane, IBM, or Xanadu
source files, tests, documentation prose, comments, error strings, wheels,
dist-info, egg-info, site-packages trees, virtual environments, or vendored
package artifacts were copied into QuantumBridge.

## Stage 9I Qiskit Experiments and Dynamics offline slices

Stage 9I adds clean-room Qiskit Experiments and Qiskit Dynamics offline
educational executable slices. They use public package names and optional
dependency detection only. They do not copy IBM, Qiskit, Qiskit Experiments,
Qiskit Dynamics, tutorial prose, UI, branding, or source code, and they do not
claim hardware calibration, production dynamics, full replacement, or production
parity.

## Stage 9J MQT Core / DDSIM / QMAP compatibility slice

Stage 9J adds clean-room MQT Core-like, DDSIM-like, and QMAP-like educational
compatibility workflows. MQT ecosystem packages remain optional upstream
dependencies for local passthrough metadata only. The native Stage 9J examples
do not require upstream MQT packages.

No MQT source code, tutorial implementation, website text, UI, branding,
wheels, dist-info, egg-info, site-packages trees, virtual environments, or
vendored package artifacts are copied into QuantumBridge. Project and package
names are used only to identify optional compatibility targets.

The native subset is not a full MQT Core, DDSIM, or QMAP replacement, not a
decision-diagram simulator parity claim, not optimal mapping software, not
production compiler/simulator/mapper parity, and not endorsed by MQT maintainers
or IBM.

## Stage 9K TorchQuantum / PyTorch-style QML compatibility slice

Stage 9K adds clean-room TorchQuantum-like educational QML workflows for a
native quantum layer, tensor/batch forward execution, deterministic toy
classifier training, optional torch tensor interop when PyTorch is installed,
and optional upstream TorchQuantum boundary metadata when installed separately.

No TorchQuantum or PyTorch source code, tutorial implementation, website text,
UI, branding, model weights, wheels, dist-info, egg-info, site-packages trees,
virtual environments, or vendored package artifacts are copied into
QuantumBridge.

The native subset is not a full TorchQuantum replacement, not a full PyTorch
replacement, not production QML training, not for high-risk automated decisions,
and not endorsed by TorchQuantum, PyTorch, IBM, or Qiskit maintainers.

## Stage 10A QOS-UQCI / Quafu backend compatibility slice

Stage 10A adds clean-room offline QOS-UQCI and Quafu-compatible backend
workflows. It uses public project/package names only and QuantumBridge-owned
job specs, payloads, mock runtime/backend execution, result schemas, examples,
and tests.

No QOS-UQCI, Quafu, or pyquafu source code, tutorial implementation, website
text, UI, branding, wheels, dist-info, egg-info, site-packages trees, virtual
environments, or vendored package artifacts are copied into QuantumBridge.

The native subset is not production QOS runtime support, not production Quafu
backend support, does not access cloud services, does not read tokens, does not
execute real hardware, and is not endorsed by QOS-UQCI or Quafu maintainers.

## Stage 10B Benchpress / Benchmarking compatibility slice

Stage 10B adds a clean-room local benchmarking adapter for QuantumBridge-owned
small educational workloads. It provides a benchmark case registry, local
runner, default suites, result schemas, JSON / Markdown reports, and optional
upstream Benchpress boundary metadata when installed separately.

No IBM, Qiskit, or Benchpress source code, tutorial implementation, website
text, UI, branding, benchmark methodology, wheels, dist-info, egg-info,
site-packages trees, virtual environments, or vendored package artifacts are
copied into QuantumBridge.

The native subset is not a full Benchpress replacement, not official benchmark
output, not a production performance ranking system, does not access cloud
services, does not read tokens, does not execute real hardware, and is not
endorsed by IBM, Qiskit, or Benchpress maintainers.

## Stage 10C QuantumBridge Studio backend API slice

Stage 10C adds a QuantumBridge-owned local backend API layer for future Studio
experiences. It provides catalog, workflow registry, input schema, local
execution, result store, export, benchmark, provenance/warning, and REST-like
local router services. It does not implement a frontend UI, does not start a
production server, and does not open network ports.

No IBM, Qiskit, PennyLane, Benchpress, QOS-UQCI, Quafu, MQT, Mitiq,
TorchQuantum, PyTorch, FastAPI, or other third-party source code, website text,
tutorial prose, UI, branding, wheels, dist-info, egg-info, site-packages trees,
or virtual environments are copied into QuantumBridge. Project and package
names are used only to identify optional compatibility targets.

This backend API slice is not an official service, not endorsed by upstream
maintainers, not a full replacement, not production API software, and does not
access cloud services, tokens, credentials, or real hardware.

## Stage 10D QuantumBridge Studio frontend prototype

Stage 10D adds a QuantumBridge-owned static local frontend prototype over the
Stage 10C backend API seed data. Its UI, layout, CSS, local data client, and
smoke checks are independently implemented for QuantumBridge.

No IBM, Qiskit, PennyLane, Benchpress, QOS-UQCI, Quafu, MQT, Mitiq,
TorchQuantum, PyTorch, FastAPI, React, Vite, Carbon, or other third-party
source code, website text, tutorial prose, UI, branding, logos, wheels,
dist-info, egg-info, site-packages trees, `node_modules`, build outputs, or
virtual environments are copied into QuantumBridge. Project and package names
are used only to identify optional compatibility targets.

This frontend prototype is not an official service, not endorsed by upstream
maintainers, not a full replacement, not production UI software, and does not
access cloud services, tokens, credentials, or real hardware.

## Stage 10E QuantumBridge Studio backend/frontend integration hardening

Stage 10E adds a QuantumBridge-owned frontend seed contract, static local
frontend client hardening, representative sample result generation, export seed
generation, benchmark seed generation, and `argparse` CLI over the local Studio
backend services.

No IBM, Qiskit, PennyLane, Benchpress, QOS-UQCI, Quafu, MQT, Mitiq,
TorchQuantum, PyTorch, FastAPI, React, Vite, Carbon, or other third-party
source code, website text, tutorial prose, UI, branding, logos, wheels,
dist-info, egg-info, site-packages trees, `node_modules`, build outputs, or
virtual environments are copied into QuantumBridge. Project and package names
are used only to identify optional compatibility targets.

This integration hardening is local-only, not an official service, not
endorsed by upstream maintainers, not a full replacement, not production UI or
production API software, and does not access cloud services, tokens,
credentials, or real hardware.

## Stage 11M Native Serialization

OpenQASM 2/3 and QPY version 13 are implemented independently from public
format specifications. Qiskit public reader/writer calls are optional
interoperability oracles, not the native implementation. No upstream source,
tests, package artifacts or environments were copied or vendored. Source
port count remains zero. Existing attribution and licenses are retained.

Public references: [OpenQASM specification](https://openqasm.com/) and
[QPY format documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qpy).
The tested subset and unsupported protocol branches are documented in the
Stage 11M report; this checkpoint does not imply complete SDK replacement.

## Stage 11L Native Transpiler

The dependency DAG, scoped compilation, algebraic gate translation,
layout/permutation routing, static scheduler and compiled QNode chain rule are
independent QuantumBridge implementations. Installed SDKs are optional public
black-box test references, not native implementation dependencies. No upstream
source/test files, package payload, environment, token or hardware job is used
or copied. The supported subset is not full SDK parity, optimal compilation,
certified physical scheduling or an endorsement by upstream maintainers.

## Stage 9B Qiskit Optimization native subset

Stage 9B adds an independently implemented educational native binary
`QuadraticProgram` subset for small optimization examples. Qiskit Optimization
remains an optional upstream dependency for passthrough execution when
installed; no upstream source files, wheels, dist-info, egg-info, site-packages
trees, virtual environments, documentation prose, or examples were copied.

The native subset is not a full Qiskit Optimization replacement, not production
optimization software, and not endorsed by IBM or Qiskit maintainers.

## Stage 9C Qiskit Algorithms native subset

Stage 9C adds independently implemented educational native VQE, QAOA-compatible
MaxCut, and Grover workflows for small examples. Qiskit Algorithms remains an
optional upstream dependency for local passthrough execution when installed; no
upstream source files, wheels, dist-info, egg-info, site-packages trees,
virtual environments, documentation prose, website text, or examples were
copied.

The native subset is not a full Qiskit Algorithms replacement, not production
algorithm software, and not endorsed by IBM or Qiskit maintainers.

## Stage 9D Qiskit Nature native chemistry subset

Stage 9D adds independently implemented educational H2 and LiH chemistry
workflows with small QuantumBridge-owned qubit Hamiltonians and local exact
diagonalization. Qiskit Nature, PySCF, and OpenFermion remain optional upstream
dependencies for local passthrough or comparison paths when installed; no
upstream source files, wheels, dist-info, egg-info, site-packages trees,
virtual environments, documentation prose, website text, or examples were
copied.

The native subset is not a full Qiskit Nature replacement, not production
quantum chemistry, not molecular design software, not a materials band-gap
workflow, and not endorsed by IBM or Qiskit maintainers.

## Stage 9E Qiskit Machine Learning native QML subset

Stage 9E adds independently implemented educational quantum-kernel, kernel
classifier, QNN forward, and QNN classifier workflows. The implementation uses
QuantumBridge-native circuits, NumPy math, deterministic toy datasets, and
optional runtime introspection of installed `qiskit-machine-learning` objects.

No Qiskit Machine Learning source code, tutorial implementation, IBM website
copy, IBM branding, model weights, training data, wheels, dist-info,
site-packages, or virtual environments are vendored. Project and package names
are used only to identify optional compatibility targets.

This subset is not a full Qiskit Machine Learning replacement, not production
machine learning software, and not intended for high-risk automated decisions.

## IBM Quantum Ecosystem clean-room parity planning

Stage 9C-Revision adds a QuantumBridge-owned ecosystem parity plan and catalog
schema. It uses public project names only for compatibility identification and
does not copy IBM Quantum Ecosystem website text, layout, images, icons,
branding, screenshots, or source. QuantumBridge is independent and does not
claim IBM, Qiskit, PennyLane, Azure Quantum, MQT, Mitiq, TorchQuantum, or other
third-party endorsement.

## TF7 Native MPS Tensor Integration

The exact low-bond TensorFlow MPS path is independently implemented from
QuantumBridge's existing physical tensor-chain, compiled native operations,
q0-first routing and local operator math. Local index splits use the elementary
factorization M=I*M or M*I, not copied simulator code, differentiated SVD
truncation or an upstream quantum executor. TensorFlow remains an optional
numerical AD dependency used through public tensor/GradientTape/ForwardAccumulator
APIs. Tests compare native dense/MPS/tangent results; installed upstream SDKs
remain optional comparison targets, not quantum execution dependencies here.
No upstream source, tests, package trees, credentials, binaries or tutorial text
are copied or vendored. This bounded analytic subset is not full MPS AD or full
independent Qiskit/PennyLane replacement.

## TF8 Exact Local Matrix Coverage

The TF MPS specialization now accepts existing QuantumBridge-owned constant
one/two-wire unitary payloads through the native matrix validator and routing.
No new third-party matrix definitions or simulator sources are imported. Native
Pauli/Hamiltonian/sparse-observable conversions are tested, not reimplemented.
Pure TensorFlow comparison diagnostics retain an actual public AD-loop failure;
they are not represented as successful native functionality or full parity.

## QC1 Public Matrix Invocation Boundaries

Public unitary/append matrix frames, local basis conversion, derived-owner
provenance and standard QPY admission are independently implemented using
existing QuantumBridge owned arrays, gate objects and execution/serialization
paths. Installed Qiskit 2.4.1/2.6.0 are optional public-behavior comparison
oracles only; their source, tests, tutorials and package trees are not copied.
Five explicitly reviewed public-entry expectations are corrected without
changing native Operator-power math or tolerances. Native matrix ownership
versus upstream aliasing, unproved compound Operator origins and complete SDK
coverage remain open; this batch does not establish full independent parity.

## QC2 Explicit Operator Arithmetic

The explicit Qiskit arithmetic convention is independently implemented on the
existing QuantumBridge Operator using its native matrix contractions, array
ownership and resource admission. Public installed SDK calls are behavioral
oracles only. No upstream source, tests, package data or numerical executor is
copied or vendored. Independent complex-matrix and mixed-radix index oracles
test the opt-in behavior separately from unchanged native defaults. Explicit
mode does not close the recorded default-API parity differences or establish
full independent Qiskit/PennyLane replacement.

## QC3 Explicit Native Arithmetic Migration Preparation

The same-class native selector, mixed-radix physical-coordinate conversion
and confirmed channel embedding callsite migration reuse QuantumBridge's
existing dense kernels, subsystem reversal and owned-array admission. Public
SDKs remain optional behavior oracles only. No upstream source, tests,
dependencies, examples or executor is copied. The AST inventory is a native
audit utility with explicitly unresolved receivers, not a claim of complete
type inference or SDK coverage. Native default arithmetic is not switched;
constructor/data-origin migration and full default parity remain open.

QC3 also qualifies binary Operator operands before shared arithmetic and
provenance propagation, using the same existing convention and owned-array
admission. Mixed-convention addition/subtraction are explicitly rejected;
no upstream arithmetic implementation or conversion heuristics are imported.

## QC4 Native State Operator Origins

Statevector and DensityMatrix operator factories retain their existing native
coordinate provenance and storage contracts. Mixed-radix conversion uses the
existing subsystem permutation, independently checked by index enumeration.
Installed Qiskit versions are optional public-call comparison oracles; no
upstream source, tests, tutorials or numerical execution implementation is
copied or vendored. Raw matrices remain literals. This scoped construction
repair does not switch default arithmetic or establish full SDK parity.

## QC4 Annotation And Channel Matrix Sources

Validated annotation matrix factories and channel construction/conversion
retain source-derived coordinate descriptions on existing native objects.
A bounded raw-unitary/controlled-annotation representation uses independently
derived register-index permutations, one owned output buffer, and existing
native resource admission. No dense permutation matrix, external array origin
registry, second executor or serialization sidecar is introduced. Installed
Qiskit versions are public-call comparison oracles only; no implementation,
tests, package contents or tutorial text are copied. Default arithmetic and
the original complete SDK objective are unchanged; full parity is incomplete.

QC4 hybrid Operator-to-Channel construction reuses the same native permutation
and channel kernels for an independent owned snapshot. Differential tests use
installed Qiskit only as an optional behavior oracle; independent tensor/index
and column-vectorized full-map calculations are authored locally. No upstream
implementation, source tree, tests or package artifacts are copied or vendored.

## QC4 Gate And Clifford Operator Sources

Native Clifford-to-Operator construction preserves the existing tableau
synthesis and its matrix phase, carrying only admitted finite coordinate
origin from proved owners. Projective Clifford comparisons are separated
from exact-phase Operator and transport checks. Public instruction export
reuses the existing owned UnitaryGate factory. Independent index-embedding
oracles and installed Qiskit comparisons are test-only; no upstream synthesis
code, tests, source trees or package artifacts are copied or vendored.

## QPY V13 Parameter Replay And Identity

Recursive expression markers, gradient replay and parameter-identity admission
are independently implemented in the existing QuantumBridge v13 reader from
the [published v13 format](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qpy).
Complete binary fixtures and analytic derivative oracles are authored locally;
installed Qiskit 2.4.1/2.6.0 calls are behavioral comparison oracles only. No
upstream parser, symbolic engine, source, tests or package artifacts are copied.
Observed upstream annotation-index and version-specific malformed-expression
failures remain explicit. Substitution replay and full QPY/SDK parity are not
established by this reader repair.

The v13 substitution reader independently decodes the published name-keyed
variable-length mapping, preserving validated native UUID ownership and
simultaneous native AST replacement. Retained formal identities are owned
context on the existing expression, separate from mathematical operands.
When needed, the existing writer projects that context into ordinary v13
arithmetic records, as independently verified against installed public readers;
no private QPY records, SDK source, symbolic executor or package data are used.
QuantumBridge IR carries explicit native context identities, not a QPY sidecar.
Reader capability does not establish API-call-history writer or full SDK parity.

The constant-gradient scalar return boundary is independently implemented on
the existing native calculus, using documented public ParameterExpression
behavior and actual installed 2.4.1/2.6.0 calls as ABI oracles. Internal repeated
derivatives keep the same bounded AST engine. Frozen scalar bindings transport
owned retained identities independently of mathematical dependencies, with
existing TensorFlow AD and resource accounting; no SDK implementation, private
gradient rule or second symbolic graph is imported. An observed Qiskit 2.4.1
trigonometric derivative discrepancy remains a genuine failing public-oracle
comparison, not a native correctness expectation or a new skip/xfail.

Managed scalar capture additionally reuses the existing bounded GraphDef data
scan for read-only occurrence provenance. Only admitted static layouts, input
positions, ordered native operations and exact float64 literal bits participate
in the proof. It emits no AST, executes/imports no graph, creates no identities
and does not expand the normal scalar-lowering capability whitelist. Unsupported
source proofs are explicit refusals, not successful SDK parity claims.
