"""
Author: weiguo_ma
Time: 07.12.2024
Contact: weiguo.m@iphy.ac.cn
"""
import sys
from copy import deepcopy
from typing import Optional, List, AnyStr, Union

import numpy as np
from FakeSampler import FakeSampler
from quafu import QuantumCircuit, simulate
from qutip import tensor, identity, Qobj, sigmax, sigmay, sigmaz, expect
from tqdm import tqdm


def circuit2densityMatrix() -> (QuantumCircuit, np.ndarray):
    qn = 2

    qc = QuantumCircuit(qn)
    qc.h(0)
    qc.cnot(0, 1)

    _outCirc = deepcopy(qc)

    qc.draw_circuit()
    measures, cbits = list(range(qn)), list(range(qn))
    qc.measure(measures, cbits)

    _simulate_res = simulate(qc)
    _density_matrix = _simulate_res.calc_density_matrix()
    return _outCirc, _density_matrix


def fakeSampling_circ(quantumCircuit: Optional[QuantumCircuit],
                      measure_orientation: Optional[List[AnyStr]] = None,
                      draw: bool = False):
    def measureX(_quantumCircuit: QuantumCircuit, qubits: List) -> QuantumCircuit:
        for qubit in qubits:
            _quantumCircuit.ry(qubit, -np.pi / 2)
        return _quantumCircuit

    def measureY(_quantumCircuit: QuantumCircuit, qubits: List) -> QuantumCircuit:
        for qubit in qubits:
            _quantumCircuit.rx(qubit, np.pi / 2)
        return _quantumCircuit

    def measureZ(_quantumCircuit: QuantumCircuit, *args) -> QuantumCircuit:
        pass

    if not isinstance(quantumCircuit, QuantumCircuit):
        raise TypeError("The type of Quantum Circuit must be quafu.QuantumCircuit.")

    _measure = {
        'X': measureX,
        'Y': measureY,
        'Z': measureZ
    }

    _qc = quantumCircuit
    _measure_sequence, _mapping_cbits = list(range(_qc.num)), list(range(_qc.num))
    _measure_orientation = measure_orientation if measure_orientation is not None else ['Z'] * _qc.num

    assert len(_measure_sequence) == _qc.num and len(_measure_orientation) == _qc.num

    for _qubit in _measure_sequence:
        _measure_option = _measure.get(_measure_orientation[_qubit])
        _measure_option(_qc, [_qubit])
    _qc.measure(_measure_sequence, _mapping_cbits)

    if draw:
        _qc.draw_circuit()

    _state = list(next(iter(simulate(_qc, shots=1).counts)))
    _state_eigenValue = ['1' if _value == '0' else '-1' for _value in _state]

    return [_item for _pair in zip(_measure_orientation, _state_eigenValue) for _item in
            _pair]  # Format: ['Z', '1', 'X', '-1', ...]


def cal_Observables(dm: Union[np.ndarray, Qobj], observables: List, keep: Optional[List] = None):
    def _convert_matrix(observablesStr):
        _obs = {
            'I': identity(2),
            'X': sigmax(),
            'Y': sigmay(),
            'Z': sigmaz()
        }

        if len(observablesStr) == 1:
            return _obs.get(observablesStr)
        else:
            observablesList = observablesStr.split()
            return tensor([_obs.get(_element) for _element in observablesList])

    _systemSize = int(np.log2(dm.shape[0]))

    if isinstance(dm, np.ndarray):
        dm = Qobj(dm, dims=[[2] * _systemSize, [2] * _systemSize])
    elif not isinstance(dm, Qobj):
        raise TypeError("The type of Density Matrix must be np.ndarray or qutip.Qobj.")

    if keep is None:
        keep = list(range(_systemSize))

    expectations = [expect(_convert_matrix(observable), dm.ptrace(keep)) for observable in observables]
    return expectations


def load_write_measurements(state: Union[QuantumCircuit, np.ndarray, Qobj],
                            schemeFile: AnyStr,
                            measureFile: AnyStr):
    if isinstance(state, QuantumCircuit):
        _systemSize = state.num
        _dmBool = False
    elif isinstance(state, np.ndarray) or isinstance(state, Qobj):
        _systemSize = int(np.log2(state.shape[0]))
        _dmBool = True
    else:
        raise TypeError("The type of Quantum State should be a quantum circuit or density matrix in np.ndarray or "
                        "qutip.Qobj.")

    sampler = FakeSampler(_systemSize)
    with open(schemeFile) as readFile, open(measureFile, 'w') as writeFile:
        writeFile.write(f"{_systemSize}\n")
        print('Start to generate fake measurement results...\n')
        for _line in tqdm(readFile):
            _orientation = _line.strip().split()
            if _dmBool:
                _measurement_orientation, _state_eigenValue = sampler.fakeSampling_dm(state, _orientation)
                _zipList = [_item for _pair in zip(_measurement_orientation, _state_eigenValue) for _item in _pair]
                _result = ' '.join(map(str, _zipList))
            else:
                _executable_qc = deepcopy(state)
                _result = ' '.join(map(str, fakeSampling_circ(_executable_qc, _orientation)))
            writeFile.write(_result + '\n')
            if not _dmBool:
                del _executable_qc  # release memory


def load_write_expectations(density_matrix, observableFile: AnyStr, expectationFile: AnyStr):
    with open(observableFile) as readFile, open(expectationFile, 'w') as writeFile:
        _systemSize = int(readFile.readline().strip())
        writeFile.write(f"{_systemSize}\n")
        print('Start to calculate expectation values...\n')
        for _line in tqdm(readFile):
            _obs = _line.strip().split()[1:]
            _observables = ' '.join(_obs[0::2])
            _locationObs = list(map(int, _obs[1::2]))
            _expectation = cal_Observables(density_matrix, [_observables], _locationObs)
            print(_expectation)
            writeFile.write(' '.join(map(str, _expectation)) + '\n')


if __name__ == "__main__":
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("python GenerateFakeMeasurement -mc [scheme.txt] [measure.txt]", file=sys.stderr)
        print("    This file generates fake measurement results with given quantum state (circuit).", file=sys.stderr)
        print("    The scheme.txt is the scheme of measurement, and measure.txt is the output of result.",
              file=sys.stderr)
        print("<or>\n", file=sys.stderr)
        print("python GenerateFakeMeasurement -md [scheme.txt] [measure.txt]", file=sys.stderr)
        print("    This file generates fake measurement results with given quantum state (density matrix).",
              file=sys.stderr)
        print("    The scheme.txt is the scheme of measurement, and measure.txt is the output of result.",
              file=sys.stderr)
        print("<or>\n", file=sys.stderr)
        print("python GenerateFakeMeasurement -e [observables.txt] [expectation.txt]", file=sys.stderr)
        print("    This file calculates expectation values of observables in .txt file with a given density matrix.",
              file=sys.stderr)
        print("    The expectation.txt saves the expectation values.", file=sys.stderr)


    if len(sys.argv) != 4:
        print_usage()
        raise ValueError("No Enough Arguments.")
    else:
        circuit, density_matrix = circuit2densityMatrix()

    if sys.argv[1] == '-mc':
        load_write_measurements(circuit, sys.argv[2], sys.argv[3])
    elif sys.argv[1] == '-md':
        load_write_measurements(density_matrix, sys.argv[2], sys.argv[3])
    elif sys.argv[1] == '-e':
        load_write_expectations(density_matrix=density_matrix, observableFile=sys.argv[2], expectationFile=sys.argv[3])
    else:
        print_usage()
