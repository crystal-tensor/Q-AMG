# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Chronological real-stock hybrid research, with train-only normalization."""
import numpy as np

from .market_data import load_stock_history


def hybrid_volatility_case(*, epochs=15, seed=20261102):
    """Native differentiable two-wire QNode + torch classical positive head.

Predict next-session absolute return from two lagged returns. This is a small
research exercise, not a trading strategy or evidence of forecasting advantage.
The separate QMC phase-estimation template is not claimed differentiable.
The older published Tape API uses adjoint derivatives; queued development
builds use simulator backprop. The actual path is included in the result.
"""
    import torch
    from quantumbridge import qml

    if type(epochs) is not int or not 1 <= epochs <= 50:
        raise ValueError("epochs must be in 1..50")
    history = load_stock_history(symbols=("IBM",))
    returns = history.returns()[:, 0]
    features = np.column_stack((returns[1:-1], returns[:-2]))
    targets = np.abs(returns[2:])
    target_dates = history.dates[3:]
    split = int(len(targets)*.75)
    center = features[:split].mean(axis=0)
    scale = features[:split].std(axis=0, ddof=1)
    target_scale = float(targets[:split].mean())
    if np.any(scale == 0) or target_scale == 0:
        raise ValueError("Training history has no variation")
    x = torch.tensor((features-center)/scale, dtype=torch.float64)
    y = torch.tensor(targets/target_scale, dtype=torch.float64)
    rng = np.random.default_rng(seed)
    weights = torch.tensor(rng.normal(0, .2, 4), dtype=torch.float64, requires_grad=True)
    head = torch.tensor([.1, .5], dtype=torch.float64, requires_grad=True)

    queued = callable(getattr(qml, "qnode", None)) and callable(getattr(qml, "device", None))
    if queued:
        @qml.qnode(qml.device("default.qubit", wires=2), interface="torch", diff_method="backprop")
        def circuit(values, theta):
            qml.RY(values[0], 0)
            qml.RY(values[1], 1)
            qml.RZ(theta[0], 0)
            qml.RY(theta[1], 0)
            qml.CNOT([0, 1])
            qml.RY(theta[2], 1)
            qml.RX(theta[3], 1)
            return qml.expval(qml.Z(1))
        differentiation = "torch backprop"
    else:
        from quantumbridge import PauliZ

        def program(tape, values, theta):
            tape.ry(values[0], 0)
            tape.ry(values[1], 1)
            tape.rz(theta[0], 0)
            tape.ry(theta[1], 0)
            tape.cx(0, 1)
            tape.ry(theta[2], 1)
            tape.rx(theta[3], 1)
            return tape.expval(PauliZ(1))

        circuit = qml.QNode(program, 2, interface="torch", diff_method="adjoint")
        differentiation = "torch interface / Native Tape adjoint"

    def predict(values):
        # Multi-source broadcast backprop is not admitted by the current core.
        # Scalar QNodes preserve genuine AD without relaxing resource contracts.
        output = torch.stack([circuit(row, weights) for row in values])
        return torch.nn.functional.softplus(head[0]*output+head[1])

    optimizer = torch.optim.Adam([weights, head], lr=.04)
    trace, gradient_norms = [], []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = ((predict(x[:split])-y[:split])**2).mean()
        loss.backward()
        if not torch.isfinite(weights.grad).all() or not torch.isfinite(head.grad).all():
            raise RuntimeError("Hybrid model produced nonfinite derivatives")
        gradient_norms.append(float(weights.grad.detach().norm()))
        trace.append(float(loss.detach())*target_scale**2)
        optimizer.step()
    predicted = predict(x[split:]).detach().numpy()*target_scale
    validation = targets[split:]
    return dict(backend=f"QuantumBridge native QNode / {differentiation} + Adam", hardware_jobs=0,
                qubits=2, epochs=epochs, source=history.provenance,
                train_dates=[target_dates[0], target_dates[split-1]],
                validation_dates=[target_dates[split], target_dates[-1]],
                dates=list(target_dates[split:]), predictions=predicted.tolist(),
                actual=validation.tolist(), train_mse=trace,
                circuit_gradient_norms=gradient_norms, weights=weights.detach().tolist(),
                classical_head=head.detach().tolist(), feature_center=center.tolist(),
                feature_scale=scale.tolist(), target_scale=target_scale,
                validation_mse=float(np.mean((predicted-validation)**2)),
                baseline_mse=float(np.mean((target_scale-validation)**2)),
                note="Chronological split; normalization and constant baseline use training observations only. No hyperparameter search on validation, no quantum-advantage claim.")
