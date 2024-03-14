from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

es_problem = driver.run()
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter

converter = QubitConverter(JordanWignerMapper())
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

numpy_solver = NumPyMinimumEigensolver()
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import VQEUCCFactory
from qiskit_nature.second_q.circuit.library import UCCSD

vqe_solver = VQEUCCFactory(Estimator(), UCCSD(), SLSQP())
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import TwoLocal

tl_circuit = TwoLocal(
    rotation_blocks=["h", "rx"],
    entanglement_blocks="cz",
    entanglement="full",
    reps=2,
    parameter_prefix="y",
)

another_solver = VQE(Estimator(), tl_circuit, SLSQP())
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

calc = GroundStateEigensolver(converter, vqe_solver)
res = calc.solve(es_problem)
print(res)
