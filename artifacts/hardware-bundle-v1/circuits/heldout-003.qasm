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
rz(-2.756426814118525) q[0];
rz(2.2501443380559385) q[1];
rz(0.9000577352223754) q[2];
rzz(-2.081383512701743) q[0], q[1];
rzz(-0.11250721690279693) q[0], q[2];
rx(0.44207344705456253) q[0];
rx(0.44207344705456253) q[1];
rx(0.44207344705456253) q[2];
barrier q[0], q[1], q[2];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
