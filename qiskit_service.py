from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel="ibm_quantum",
    token="IBM_QUANTUM_API_TOKEN"
)