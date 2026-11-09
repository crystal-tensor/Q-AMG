OPENQASM 3.0;
include "stdgates.inc";
gate rzz(p0) _gate_q_0, _gate_q_1 {
  cx _gate_q_0, _gate_q_1;
  rz(p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
}
bit[4] meas;
qubit[4] q;
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.0279632092711171) q[0];
rz(-0.16777925562670262) q[1];
rz(-0.6711170225068105) q[2];
rz(-0.25166888344005395) q[3];
rzz(-4.362260646294268) q[0], q[1];
rzz(-0.1957424648978197) q[0], q[2];
rzz(-0.08388962781335131) q[2], q[3];
rx(3.878236786610759) q[0];
rx(3.878236786610759) q[1];
rx(3.878236786610759) q[2];
rx(3.878236786610759) q[3];
barrier q[0], q[1], q[2], q[3];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
