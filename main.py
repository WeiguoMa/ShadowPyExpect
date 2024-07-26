import numpy as np
from qutip import ket2dm, basis, tensor

from src.python.z2_model import Z2Model

init_dm = ket2dm(
    tensor(
        [
            basis(2, 0),
            (basis(2, 0) + basis(2, 1)) / np.sqrt(2),
            (basis(2, 0) + basis(2, 1)) / np.sqrt(2),
            basis(2, 1)
        ]
    )
)

z2Properties = Z2Model(system_size=4, whole_time=200, time_step=50)

expectation_qutip = z2Properties.calculate_expectation(init_state=init_dm)

expectations_CS = z2Properties.calculate_expectation(init_state=init_dm, cs=True, measure_times=300)

# import pickle
#
# with open('expectations_CS.pkl', 'wb') as f:
#     pickle.dump(expectations_CS, f)

Gauss0, Gauss1 = [], []
magnetization = []
main = []
for dm_key, cs_expectation in expectations_CS.items():
    main.append(
        (cs_expectation['XXX-[0, 1, 2]'] - cs_expectation['XXY-[0, 1, 2]']
         + cs_expectation['YXX-[0, 1, 2]'] + cs_expectation['YXY-[0, 1, 2]']
         + cs_expectation['XXX-[2, 3, 0]'] - cs_expectation['XXY-[2, 3, 0]']
         + cs_expectation['YXX-[2, 3, 0]'] + cs_expectation['YXY-[2, 3, 0]']

         + cs_expectation['XXX-[0, 1, 2]'] + cs_expectation['XXY-[0, 1, 2]']
         - cs_expectation['YXX-[0, 1, 2]'] + cs_expectation['YXY-[0, 1, 2]']
         + cs_expectation['XXX-[2, 3, 0]'] + cs_expectation['XXY-[2, 3, 0]']
         - cs_expectation['YXX-[2, 3, 0]'] + cs_expectation['YXY-[2, 3, 0]']) * 3 / 2
        + cs_expectation['Z-[0]'] - cs_expectation['Z-[2]']
        + (cs_expectation['Z-[1]'] + cs_expectation['Z-[3]']) * 3 / 2
    )

    magnetization.append(
        cs_expectation['Z-[0]'] + cs_expectation['Z-[2]']
    )

    Gauss0.append(
        cs_expectation['ZZZ-[3, 0, 1]']
    )

    Gauss1.append(
        cs_expectation['ZZZ-[1, 2, 3]']
    )

# print("Conservation Main Hamiltonian - Expectation:")
# print(main)
# print('\n')
print("Conservation Magnetization Hamiltonian - Expectation:")
print(magnetization)
magnetization = np.array(magnetization)
print(
    f"MeanValue: {np.mean(magnetization)} and the Standard Deviation: {np.std(magnetization)}, length: {len(magnetization)}")
print('\n')
# print("Conservation Gauss Law Hamiltonian - Expectation:")
# print("Expectation of Gauss Law Hamiltonian 1:")
# print(Gauss0)
# print("Expectation of Gauss Law Hamiltonian 2:")
# print(Gauss1)
