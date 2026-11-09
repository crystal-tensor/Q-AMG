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
rz(-2.117570951741861) q[0];
rz(2.887596752375265) q[1];
rz(-0.08250276435357899) q[2];
rz(0.137504607255965) q[3];
rzz(-2.447582009156177) q[0], q[1];
rzz(-0.11000368580477202) q[0], q[2];
rzz(-0.08250276435357899) q[2], q[3];
rx(0.46347506388981674) q[0];
rx(0.46347506388981674) q[1];
rx(0.46347506388981674) q[2];
rx(0.46347506388981674) q[3];
barrier q[0], q[1], q[2], q[3];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
