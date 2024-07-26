from typing import Optional, List, Dict, Iterable

import numpy as np
from qutip import mesolve, Options
from qutip import tensor, Qobj, sigmax, sigmay, sigmaz, qeye

from .abstract_properties import AbstractModelProperties
from .observables_generator import generate_observables

__all__ = [
    'Z2Model'
]


class Z2:
    def __init__(self, system_size: int,
                 m: float = 1.0,
                 a: Optional[float] = None,
                 e: Optional[float] = None,
                 write_observables: bool = True,
                 obs_write_file_name: Optional[str] = None):
        """
        Initialize the Z2 model.

        Args:
            system_size (int): The size of the system.
            m (float): Parameter m.
            a (Optional[float]): Parameter a. Default is m/3.
            e (Optional[float]): Parameter e. Default is 3/(2*m).
            write_observables (bool): Flag to write observables.
            obs_write_file_name (Optional[str]): File name for writing observables.
        """
        self.q_number = system_size
        assert self.q_number % 2 == 0

        self.m = m
        self.a = self.m / 3 if a is None else a
        self.e = 3 / (2 * self.m) if e is None else e

        self.x, self.y, self.z, self.i = sigmax(), sigmay(), sigmaz(), qeye(2)
        self.sigma_plus, self.sigma_minus = (self.x + 1j * sigmay()) / 2, (self.x - 1j * sigmay()) / 2

        self.main_ham = self.system_ham()
        self.gauss_law_ham = self.gauss_law()
        self.magnetization_ham = self.magnetization()

        self.observables_cs: Dict = self.get_observables_cs(write=write_observables,
                                                            write_file_name=obs_write_file_name)

    def construct_operator(self, ops: Optional[List[Qobj]] = None, pos: Optional[List[int]] = None) -> Qobj:
        """
        Construct an operator for the Z2 model.

        Args:
            ops (Optional[List[Qobj]]): List of operators.
            pos (Optional[List[int]]): List of positions.

        Returns:
            Qobj: The constructed operator.
        """
        op_list = [self.i] * self.q_number

        if ops is not None and pos is not None:
            if pos[0] == -1:  # Periodic boundary condition.
                pos[0] = self.q_number - 1
            if pos[-1] == self.q_number:
                pos[-1] = 0

            for op, p in zip(ops, pos):
                op_list[p] = op

        return tensor(op_list)

    def system_ham(self) -> Qobj:
        """
        Construct the system Hamiltonian for the Z2 model.

        Returns:
            Qobj: The system Hamiltonian.
        """
        H = Qobj(np.zeros((2 ** self.q_number, 2 ** self.q_number)), dims=[[2] * self.q_number, [2] * self.q_number])

        for i in range(self.q_number // 2):
            operator = self.construct_operator(
                [self.sigma_plus, self.x, self.sigma_minus], [2 * i, 2 * i + 1, 2 * i + 2]
            )

            H += (1 / (2 * self.a) * (operator + operator.dag())
                  + self.m * ((-1) ** i) / 2 * (
                          self.construct_operator() + self.construct_operator([self.z], [2 * i])
                  )
                  + self.e * self.construct_operator([self.z], [2 * i + 1]))
        return H

    def magnetization(self) -> Qobj:
        """
        Construct the magnetization Hamiltonian for the Z2 model.

        Returns:
            Qobj: The magnetization Hamiltonian.
        """
        H = Qobj(np.zeros((2 ** self.q_number, 2 ** self.q_number)), dims=[[2] * self.q_number, [2] * self.q_number])
        for i in range(self.q_number // 2):
            H += self.construct_operator([self.z], [2 * i])
        return H

    def gauss_law(self) -> List[Qobj]:
        """
        Construct the Gauss law observables for the Z2 model.

        Returns:
            List[Qobj]: The Gauss law observables.
        """
        obs = [
            self.construct_operator([self.z] * 3, [2 * i - 1, 2 * i, 2 * i + 1])
            for i in range(self.q_number // 2)
        ]
        return obs

    def get_observables_cs(self, write: bool = True, write_file_name: Optional[str] = None) -> Dict:
        """
        Get the observables for Classical Shadow.

        Args:
            write (bool): Flag to write observables.
            write_file_name (Optional[str]): File name for writing observables.

        Returns:
            Dict: Dictionary of observables and their positions.
        """
        observables_part0, positions_part0 = generate_observables(
            self.q_number, ['+', 'X', '-'],
            repeat_period=[[2 * i, 2 * i + 1, 2 * i + 2] for i in range(self.q_number // 2)],
            write=write,
            write_file_name=write_file_name
        )
        observables_part1, positions_part1 = generate_observables(
            self.q_number, ['Z'],
            repeat_period=[[i] for i in range(self.q_number)],
            write=write,
            write_file_name=write_file_name,
            write_mode='a'
        )
        observables_part2, positions_part2 = generate_observables(
            self.q_number, ['Z', 'Z', 'Z'],
            repeat_period=[[2 * i - 1, 2 * i, 2 * i + 1] for i in range(self.q_number // 2)],
            write=write,
            write_file_name=write_file_name,
            write_mode='a'
        )

        observables: List = observables_part0 + observables_part1 + observables_part2
        positions: List = positions_part0 + positions_part1 + positions_part2

        observables_cs: Dict = {  # Format: ['XXX', 'XXX', ...], [[0, 1, 2], [2, 3, 0], ...]
            'system_size': self.q_number,
            'k_local': [len(ele) for ele in observables],
            'observables': observables,
            'positions': positions
        }

        return observables_cs


class Z2Model(AbstractModelProperties):
    def __init__(self,
                 system_size: int,
                 whole_time: float,
                 time_step: int,
                 observables_cs: Optional[Dict] = None,
                 obs_write: bool = False,
                 obs_write_file_name: Optional[str] = None):
        super().__init__(system_size=system_size,
                         observables_cs=observables_cs,
                         obs_write=obs_write,
                         obs_write_file_name=obs_write_file_name)

        self.time, self.steps = whole_time, time_step
        self.time_slice = np.linspace(0, self.time, self.steps)

        self.expectation_series_qutip = None

    def create_model(self):
        return Z2(self.q_number,
                  m=1,
                  write_observables=self.model_obs_write,
                  obs_write_file_name=self.model_obs_write_file_name)

    def get_density_matrix(self, init_state: Optional[Qobj] = None, system_hamiltonian: Optional[Qobj] = None) -> List[
        Qobj]:
        """
        Get the density matrix of the system over time.

        Args:
            init_state (Optional[Qobj]): Initial state of the system.
            system_hamiltonian (Optional[Qobj]): Hamiltonian of the system.

        Returns:
            List[Qobj]: List of density matrices over time.
        """
        opts = Options(store_states=True)

        system_hamiltonian = self.model.main_ham if system_hamiltonian is None else system_hamiltonian
        result = mesolve(system_hamiltonian, self.init_state, self.time_slice, options=opts)

        return result.states

    def calculate_expectation(self, init_state: Optional[Qobj] = None,
                              cs: bool = False,
                              system_hamiltonian: Optional[Qobj] = None,
                              time_slice: Optional[Iterable] = None,
                              observables: Optional[List[Qobj]] = None,
                              measure_times: Optional[int] = None,
                              derandom_scheme: bool = True,
                              measure_scheme_file_name: Optional[str] = None,
                              measure_outcome_file_name: Optional[str] = None) -> Dict:
        """
        Calculate the expectation values of observables.

        Args:
            init_state (Optional[Qobj]): Initial state of the system.
            cs (bool): Flag to use Classical Shadow method.
            system_hamiltonian (Optional[Qobj]): Hamiltonian of the system.
            time_slice (Optional[Iterable]): Time slices for calculation.
            observables (Optional[List[Qobj]]): List of observables.
            measure_times (Optional[int]): Number of measurement times.
            derandom_scheme (bool): Flag to use derandomized scheme.
            measure_scheme_file_name (Optional[str]): File name for the measurement scheme.
            measure_outcome_file_name (Optional[str]): File name for the measurement outcomes.

        Returns:
            Dict[str, List[float]]: Dictionary of expectation values.
        """
        if init_state is None:
            raise NotImplementedError("Expectations for arbitrary quantum states are not supported yet.")
        else:
            if not cs:
                print('------- Calculating Expectations of Observables with Qutip -------')
                if not isinstance(observables, list) and observables is not None:
                    raise TypeError(f'Type of observables must be List[Qobj], current type is {type(observables)}.')

                observables = [] if observables is None else observables
                _observables = (
                        [self.model.main_ham, self.model.magnetization_ham]
                        + [_ham for _ham in self.model.gauss_law_ham]
                        + observables
                )
                system_hamiltonian = self.model.main_ham if system_hamiltonian is None else system_hamiltonian
                time_slice = self.time_slice if time_slice is None else time_slice

                self.expectation_series_qutip = self._calculate_expectation_qutip(
                    system_hamiltonian=system_hamiltonian, init_state=init_state,
                    time_slice=time_slice, observables=_observables
                )
                return self.expectation_series_qutip
            else:
                if not isinstance(measure_times, int):
                    raise ValueError(f"MeasureTimes must be a finite integer, not {measure_times}.")

                print('------- Calculating Expectations of Observables with Classical Shadow -------')
                expectation_series_classical_shadow = self._calculate_expectation_classical_shadow(
                    measurement_times=measure_times,
                    derandom_scheme=derandom_scheme
                )
                return expectation_series_classical_shadow
