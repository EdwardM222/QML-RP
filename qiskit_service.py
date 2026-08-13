from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeKingston, FakeMarrakesh

service = QiskitRuntimeService(token="YOUR_IBM_QUANTUM_API_TOKEN")

backend = FakeFez()
backend.refresh(
    service=service,
    use_fractional_gates=True,
)
backend = FakeFez()

backend = FakeKingston()
backend.refresh(
    service=service,
    use_fractional_gates=True,
)
backend = FakeKingston()

backend = FakeMarrakesh()
backend.refresh(
    service=service,
    use_fractional_gates=True,
)
backend = FakeMarrakesh()

if __name__ == "__main__":
    print(service.backends())