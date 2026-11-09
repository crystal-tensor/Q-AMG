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
rz(-1.4236490138853921) q[0];
rz(2.756426814118525) q[1];
rz(-0.36348485460903623) q[2];
rzz(-2.5141035777125005) q[0], q[1];
rzz(-0.12116161820301209) q[0], q[2];
rx(0.44207344705456253) q[0];
rx(0.44207344705456253) q[1];
rx(0.44207344705456253) q[2];
barrier q[0], q[1], q[2];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
