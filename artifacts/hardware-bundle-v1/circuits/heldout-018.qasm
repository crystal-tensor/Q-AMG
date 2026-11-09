OPENQASM 3.0;
include "stdgates.inc";
gate rzz(p0) _gate_q_0, _gate_q_1 {
  cx _gate_q_0, _gate_q_1;
  rz(p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
}
bit[3] meas;
qubit[3] q;
h q[0];
h q[1];
h q[2];
rz(0.2892624067737566) q[0];
rz(-0.09381483462932648) q[1];
rz(0.11726854328665808) q[2];
rzz(-1.0632347924657) q[0], q[1];
rzz(-0.054725320200440436) q[0], q[2];
rx(2.350322876149115) q[0];
rx(2.350322876149115) q[1];
rx(2.350322876149115) q[2];
barrier q[0], q[1], q[2];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
