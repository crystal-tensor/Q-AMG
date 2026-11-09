# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Actual bounded amplitude-estimation workflows, not synthetic finance quotes."""
from copy import deepcopy
from math import exp, log, sqrt

import numpy as np

from quantumbridge import Circuit, EstimationProblem, MaximumLikelihoodAmplitudeEstimation
from quantumbridge.primitives import StatevectorSampler
from .kernels import prepare_real_state, risk_measures
from .market_data import load_stock_history


def _distribution(probabilities, values):
    if np.iscomplexobj(probabilities) or np.iscomplexobj(values):
        raise ValueError("probabilities and payoffs must be real; imaginary parts are not discarded")
    p, v = np.asarray(probabilities, dtype=float), np.asarray(values, dtype=float)
    if (p.ndim != 1 or len(p) not in (2, 4, 8) or v.shape != p.shape
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(v))
            or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("2, 4 or 8 normalized probabilities and finite matching values required")
    return p, v


def expectation_problem(probabilities, values):
    """Encode signed values by an affine shift; no clipping of the payoff."""
    p, values = _distribution(probabilities, values)
    lower = float(values.min())
    scale = float(np.ptp(values)) or 1.
    payoff = (values-lower)/scale
    amplitudes = np.stack((np.sqrt(p*(1-payoff)), np.sqrt(p*payoff)), axis=1).reshape(-1)
    prepared = prepare_real_state(amplitudes)
    circuit = Circuit(prepared.num_qubits, name="real-stock-expectation-A")
    circuit.operations.extend(deepcopy(prepared.operations))
    # S0 = I-2|0><0|. An exact Walsh phase expansion avoids opaque
    # multi-controlled gates in the financial hardware-review artifact.
    width = circuit.num_qubits
    grover = Circuit(width).z(width-1).compose(circuit.inverse())
    for mask in range(1, 1 << width):
        wires = [w for w in range(width) if mask & (1 << (width-1-w))]
        for wire in wires[:-1]:
            grover.cx(wire, wires[-1])
        grover.rz(-2*np.pi/(1 << width), wires[-1])
        for wire in reversed(wires[:-1]):
            grover.cx(wire, wires[-1])
    grover.global_phase = np.pi/(1 << width)+np.pi
    grover.compose(circuit, inplace=True)
    problem = EstimationProblem(circuit, [width-1], grover_operator=grover,
                                post_processing=lambda a: lower+scale*a)
    return problem, lower, scale


def estimate_expectation(probabilities, values, *, shots=1024, seed=20261102):
    """Run A,Q A,Q^2 A,Q^4 A using real native Sampler V2 counts and MLAE."""
    p, values = _distribution(probabilities, values)
    if type(shots) is not int or not 32 <= shots <= 8192:
        raise ValueError("shots must be in 32..8192")
    problem, lower, scale = expectation_problem(p, values)
    sampler = StatevectorSampler(default_shots=shots, seed=np.random.default_rng(seed))
    algorithm = MaximumLikelihoodAmplitudeEstimation([0, 1, 2, 4], sampler=sampler)
    result = algorithm.estimate(problem, shots=shots)
    counts = [dict(c) for c in result.circuit_results]
    interval = algorithm.compute_confidence_interval(result, .05, kind="likelihood_ratio",
                                                     apply_post_processing=True)
    return dict(backend="QuantumBridge.StatevectorSampler V2 / MLAE", hardware_jobs=0,
                qubits=problem.state_preparation.num_qubits, shots_per_circuit=shots,
                schedule=list(result.evaluation_schedule), counts=counts,
                good_counts=list(result.good_counts), exact_discrete=float(p@values),
                estimate=float(result.estimation_processed),
                unamplified_estimate=lower+scale*result.good_counts[0]/shots,
                confidence_interval=list(map(float, interval)),
                confidence_note="95% asymptotic likelihood-ratio envelope, not finite-sample coverage guarantee; model/discretization/hardware errors excluded",
                grover_queries=int(result.num_oracle_queries),
                preparation_queries=shots*sum(2*k+1 for k in result.evaluation_schedule),
                total_shots=shots*len(counts),
                preparation=problem.state_preparation,
                circuits=algorithm.construct_circuits(problem),
                normalization={"offset": lower, "scale": scale})


def pennylane_qmc(probabilities, values, *, estimation_wires=5):
    """Execute installed PennyLane's genuine QuantumMonteCarlo on default.qubit.

Its Q operator has phase convention mu=(1-cos(pi*theta))/2. This is a
coarse phase-grid estimate, not the native MLAE output or a hardware result.
"""
    import pennylane as qml

    p, values = _distribution(probabilities, values)
    if type(estimation_wires) is not int or not 2 <= estimation_wires <= 6:
        raise ValueError("estimation_wires must be in 2..6")
    # This template's squared reflection phase aliases mu and 1-mu. Keep the
    # encoded expectation in [0, 1/2] so conjugate-phase folding is unambiguous.
    lower, scale = float(values.min()), 2*float(np.ptp(values)) or 1.
    f = (values-lower)/scale
    target = len(p).bit_length()
    estimate_wires = list(range(target, target+estimation_wires))
    device = qml.device("default.qubit", wires=target+estimation_wires)

    @qml.qnode(device)
    def circuit():
        qml.QuantumMonteCarlo(p, lambda i: f[i], target_wires=range(target),
                              estimation_wires=estimate_wires)
        return qml.probs(wires=estimate_wires)

    probabilities = np.asarray(circuit(), dtype=float)
    # Fold both conjugate phase peaks before selecting the modal grid point.
    size = len(probabilities)
    folded = probabilities[:size//2+1].copy()
    for index in range(1, size//2):
        folded[index] += probabilities[-index]
    phase = int(np.argmax(folded))/size
    grid_amplitude = float((1-np.cos(np.pi*phase))/2)
    estimate = lower+scale*grid_amplitude
    # The full outcome distribution is retained: low phase resolution is visible.
    return dict(backend="PennyLane.default.qubit / QuantumMonteCarlo", version=qml.__version__,
                estimate=float(estimate), phase=phase, probabilities=probabilities.tolist(),
                qubits=target+estimation_wires, estimation_wires=estimation_wires,
                exact_discrete=float(p@values), hardware_jobs=0,
                grid_amplitude=grid_amplitude,
                normalization={"offset": lower, "scale": scale, "range": "[0, 1/2] to avoid reflection phase alias"})


def pricing_case(*, strike_ratio=1., rate=0.03, maturity_days=30, shots=1024, seed=20261102):
    """IBM historical calibration; European model prices, never option quotes."""
    from scipy.stats import norm

    if (not np.all(np.isfinite([strike_ratio, rate, maturity_days]))
            or not 0.5 <= strike_ratio <= 1.5 or not -0.1 <= rate <= 0.3
            or type(maturity_days) is not int or not 1 <= maturity_days <= 252):
        raise ValueError("bounded strike, rate and integer maturity required")
    history = load_stock_history(symbols=("IBM",))
    spot = float(history.prices[-1, 0])
    log_returns = np.diff(np.log(history.prices[:, 0]))
    sigma = float(log_returns.std(ddof=1)*sqrt(252))
    maturity = maturity_days/252
    strike = spot*strike_ratio
    z = norm.ppf((np.arange(8)+.5)/8)
    terminal = spot*np.exp((rate-.5*sigma*sigma)*maturity+sigma*sqrt(maturity)*z)
    discount = exp(-rate*maturity)
    call = discount*np.maximum(terminal-strike, 0.)
    put = discount*np.maximum(strike-terminal, 0.)
    p = np.full(8, 1/8)
    d1 = (log(spot/strike)+(rate+.5*sigma*sigma)*maturity)/(sigma*sqrt(maturity))
    d2 = d1-sigma*sqrt(maturity)
    bs_call = spot*norm.cdf(d1)-strike*discount*norm.cdf(d2)
    bs_put = strike*discount*norm.cdf(-d2)-spot*norm.cdf(-d1)
    return dict(source=history.provenance, dates=[history.dates[0], history.dates[-1]],
                spot=spot, strike=strike, volatility=sigma, rate=rate,
                maturity_days=maturity_days, probability_measure="Risk-neutral Q; GBM, zero dividend yield assumed",
                scenario_prices=terminal.tolist(), probabilities=p.tolist(),
                call_payoffs=call.tolist(), put_payoffs=put.tolist(),
                black_scholes={"call": float(bs_call), "put": float(bs_put)},
                call=estimate_expectation(p, call, shots=shots, seed=seed),
                put=estimate_expectation(p, put, shots=shots, seed=seed+1))


def risk_case(*, confidence=.95, notional=1000., shots=1024, seed=20261102):
    """One-day empirical IBM risk: binning error and shot error kept separate."""
    if (not np.all(np.isfinite([confidence, notional])) or not .5 <= confidence <= .99
            or not 0 < notional <= 1e9):
        raise ValueError("confidence .5..99 and positive bounded notional required")
    history = load_stock_history(symbols=("IBM",))
    losses = -notional*history.returns()[:, 0]
    reference = risk_measures(losses, np.full(len(losses), 1/len(losses)), confidence)
    bins = np.array_split(np.sort(losses), 8)
    values = np.array([group.mean() for group in bins])
    p = np.array([len(group)/len(losses) for group in bins])
    binned = risk_measures(values, p, confidence)
    # Preserve the raw-history VaR threshold. Tail-excess bin means preserve
    # the historical CVaR exactly before amplitude-estimation shot noise.
    var = reference["VaR"]
    excess = np.array([np.maximum(group-var, 0).mean() for group in bins])
    tail = estimate_expectation(p, excess, shots=shots, seed=seed+1)
    return dict(source=history.provenance, dates=[history.dates[0], history.dates[-1]],
                notional=notional, confidence=confidence, probability_measure="Empirical historical P; not risk-neutral pricing",
                losses=losses.tolist(), bin_values=values.tolist(), probabilities=p.tolist(),
                raw_reference=reference, centroid_binned_reference=binned,
                mean=estimate_expectation(p, values, shots=shots, seed=seed),
                excess_values=excess.tolist(), tail=tail,
                cvar_estimate=var+tail["estimate"]/(1-confidence),
                cvar_interval=[var+a/(1-confidence) for a in tail["confidence_interval"]],
                note="VaR threshold is computed classically from observed history, not claimed quantum quantile search. Tail conditional means retain excess expectation; finite historical sample uncertainty excluded.")
