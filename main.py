import numpy as np
from qutip import ket2dm, basis, tensor
from src.python.z2_model import Z2Model

def main():
    """
    Main function to demonstrate the use of Z2Model for calculating expectations.
    """

    # Initialize the density matrix for a 4-qubit system
    init_dm = initialize_density_matrix()

    # Create an instance of the Z2Model
    z2_model = Z2Model(system_size=4, whole_time=200, time_step=50)

    # Calculate expectations using QuTiP
    expectation_qutip = z2_model.calculate_expectation(init_state=init_dm)

    # Calculate expectations using classical simulation (CS)
    expectations_cs = z2_model.calculate_expectation(init_state=init_dm, cs=True, measure_times=300)

    # Analyze and print the results
    analyze_results(expectations_cs)

def initialize_density_matrix():
    """
    Initialize the density matrix for a 4-qubit system.
    """
    # Create the tensor product of basis states
    state = tensor(
        [
            basis(2, 0),
            (basis(2, 0) + basis(2, 1)) / np.sqrt(2),
            (basis(2, 0) + basis(2, 1)) / np.sqrt(2),
            basis(2, 1)
        ]
    )

    # Convert the state to a density matrix
    return ket2dm(state)

def analyze_results(expectations_cs):
    """
    Analyze and print the results of the expectation calculations.
    """
    gauss0, gauss1 = [], []
    magnetization = []

    for dm_key, cs_expectation in expectations_cs.items():
        # Calculate magnetization
        magnetization.append(
            cs_expectation['Z-[0]'] + cs_expectation['Z-[2]']
        )

        # Calculate Gauss law expectations
        gauss0.append(
            cs_expectation['ZZZ-[3, 0, 1]']
        )

        gauss1.append(
            cs_expectation['ZZZ-[1, 2, 3]']
        )

    # Print results
    print_results(magnetization, gauss0, gauss1)

def print_results(magnetization, gauss0, gauss1):
    """
    Print the results of the magnetization and Gauss law expectations.
    """
    print('\n')
    print("Conservation Magnetization Hamiltonian - Expectation:")
    print(magnetization)
    magnetization = np.array(magnetization)
    print(
        f"Mean Value: {np.mean(magnetization)} and the Standard Deviation: {np.std(magnetization)}, length: {len(magnetization)}"
    )
    print('\n')
    print("Conservation Gauss Law Hamiltonian - Expectation:")
    print("Expectation of Gauss Law Hamiltonian 1:")
    print(gauss0)
    gauss0 = np.array(gauss0)
    print(
        f"Mean Value: {np.mean(gauss0)} and the Standard Deviation: {np.std(gauss0)}, length: {len(gauss0)}"
    )
    print('\n')
    print("Expectation of Gauss Law Hamiltonian 2:")
    print(gauss1)
    gauss1 = np.array(gauss1)
    print(
        f"Mean Value: {np.mean(gauss1)} and the Standard Deviation: {np.std(gauss1)}, length: {len(gauss1)}"
    )


if __name__ == "__main__":
    main()
