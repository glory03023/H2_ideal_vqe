from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver, Psi4Driver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import VQEUCCFactory
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import TwoLocal
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Define the H-H distances to sample
distances = np.arange(0.1, 4.0, 0.1)
molecule = 'H .0 .0 -{0}; H .0 .0 {0}'
basisNames = ["sto3g", "321g"]

# Initialize arrays to store distances and energies
dissociation_curve = []

# Loop over basis
for basis in basisNames:
    cur_curve = []
    # Loop over distances
    for distance in distances:
        # Define molecule and driver
        driver = PySCFDriver(
            atom=molecule.format(distance/2),
            basis=basis,
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )

        es_problem = driver.run()

        converter = QubitConverter(JordanWignerMapper())

        vqe_solver = VQEUCCFactory(Estimator(), UCCSD(), SLSQP())

        tl_circuit = TwoLocal(
            rotation_blocks=["h", "rx"],
            entanglement_blocks="cz",
            entanglement="full",
            reps=2,
            parameter_prefix="y",
        )

        calc = GroundStateEigensolver(converter, vqe_solver)
        result = calc.solve(es_problem)
        # Calculate energy
        energy = result.raw_result.optimal_value
        cur_curve.append((distance, energy))
    dissociation_curve.append(cur_curve)

# Print the dissociation curve
with open("report.txt", "w") as file:
    i = 0
    # Loop over basis
    for basis in basisNames:    
        print(f"Dissociation Curve of {basis} (H-H distance vs. Energy)", file=file)
        print("-------------------------------------------", file=file)
        print("H-H Distance (Å)   Energy (Hartree)", file=file)
        for distance, energy in dissociation_curve[i]:
            print(f"{distance:.2f}               {energy:.6f}", file=file)
        i = i + 1

for i in range(len(basisNames)):
    x, y = zip(*dissociation_curve[i])
    plt.plot(x, y, marker='o', label=basisNames[i])  # 'o' for circular markers
plt.xlabel('H-H Distance (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('H2 Ground State Energy')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('report.png')  # Save the plot to a file