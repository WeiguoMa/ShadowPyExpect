from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Iterable

from numpy import ndarray
from qutip import Qobj, Options, mesolve
from tqdm import tqdm

from .FakeSampler import FakeSampler
from .class_cpp2python import MeasureScheme, ClassicalShadow


class AbstractModelProperties(ABC):
    def __init__(self,
                 system_size: int,
                 observables_cs: Optional[Dict] = None,
                 density_matrix_series: Optional[List[ndarray]] = None,
                 model: Optional = None,
                 obs_write: bool = False,
                 obs_write_file_name: Optional[str] = None):
        """
        Abstract class for saving the model properties.

        Args:
            system_size (int): The size of the system.
            observables_cs (Optional[Dict]): Observables for Classical Shadow.
            density_matrix_series (Optional[List[ndarray]]): Series of density matrices.
            model (Optional): The model instance.
            obs_write (bool): Flag to write observables.
            obs_write_file_name (Optional[str]): File name for writing observables.
        """
        self.q_number = system_size
        self.init_state = None
        self.model_obs_write = obs_write
        self.model_obs_write_file_name = obs_write_file_name
        self.model = model if model is not None else self.create_model()
        self.observables_cs = self.model.observables_cs if observables_cs is None else observables_cs

        # tools
        self.measure_scheme_generator = MeasureScheme()
        self.fake_sampler = FakeSampler(system_size=system_size)

        self.dm_time_series = density_matrix_series
        self.measure_scheme, self.measure_outcomes = None, None
        self.expectation_series_classical_shadow = None
        self.expectation_series_qutip = None

    @abstractmethod
    def create_model(self):
        """
        Creates the model instance.

        Must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def get_density_matrix(self, init_state: Optional[Qobj] = None):
        """
        Get the density matrix of a quantum system with time slices.

        This can be calculated by the 'mesolve' function in QuTiP,
        simulate quantum circuit with Trotter decomposition,
        or even from unknown quantum states.
        """
        pass

    @abstractmethod
    def calculate_expectation(self, init_state: Optional[Qobj] = None,
                              cs: bool = True,
                              system_hamiltonian: Optional[Qobj] = None,
                              time_slice: Optional[Iterable] = None,
                              observables: Optional[List[Qobj]] = None,
                              measure_times: Optional[int] = None,
                              derandom_scheme: bool = True,
                              measure_scheme_file_name: Optional[str] = None,
                              measure_outcome_file_name: Optional[str] = None):
        """
        Calculates the expectation values using QuTiP or ShadowPyExpect.

        If init_state is None, the process density matrix should be offered for calculating expectations.

        Args:
            init_state (Optional[Qobj]): Initial state of the quantum system.
            cs (bool): Flag to indicate use of Classical Shadow.
            system_hamiltonian (Optional[Qobj]): Hamiltonian of the system.
            time_slice (Optional[Iterable]): Time slices for evolution.
            observables (Optional[List[Qobj]]): List of observables.
            measure_times (Optional[int]): Number of measurement times.
            derandom_scheme (bool): Flag to indicate derandomized measurement scheme.
            measure_scheme_file_name (Optional[str]): File name for the measurement scheme.
            measure_outcome_file_name (Optional[str]): File name for the measurement outcomes.
        """
        pass

    def _calculate_expectation_qutip(self,
                                     system_hamiltonian: Qobj,
                                     init_state: Qobj,
                                     time_slice: Iterable,
                                     observables: List[Qobj]):
        opts = Options(store_states=True)
        result = mesolve(system_hamiltonian, init_state, time_slice,
                         e_ops=observables, options=opts)

        self.dm_time_series = result.states
        return result.expect

    def _generate_measure_scheme(self,
                                 measurement_times: int,
                                 derandom_scheme: bool = True) -> List[List[str]]:
        if derandom_scheme:
            measure_scheme: List = self.measure_scheme_generator.derandom_generate(
                measurement_times_per_observable=measurement_times,
                observables_cs=self.observables_cs,
            )
        else:
            measure_scheme: List = self.measure_scheme_generator.random_generate(
                total_measurement_times=measurement_times,
            )

        self.measure_scheme = measure_scheme
        return measure_scheme

    def _get_measurement_outcomes(self,
                                  measure_scheme: Optional[List] = None) -> Dict[str, Dict[str, List]]:
        if measure_scheme is None:
            raise ValueError("The measurement scheme must be provided.")
        return {
            f'DM_{i}': {  # ori: ['X', 'X', 'X', 'X'] -> outcome: [1, -1, -1, 1]
                'orientations': list(orientations),
                'outcomes': [[int(outcome) for outcome in sublist] for sublist in outcomes]  # str -> int
            }
            for i, _dm in tqdm(enumerate(self.dm_time_series))
            for orientations, outcomes in
            [zip(*[self.fake_sampler.fake_sampling_dm(_dm, scheme) for scheme in measure_scheme])]
        }

    def _calculate_expectation_classical_shadow(self,
                                                measurement_times: int,
                                                derandom_scheme: bool = True):
        """
        Args:
            measurement_times (int): The number of measurement times.
        """
        measure_scheme = self._generate_measure_scheme(measurement_times=measurement_times,
                                                       derandom_scheme=derandom_scheme)

        # Get the target unknown quantum state (Qutip/QuantumCircuit simulation/Unknown Capture)
        if self.dm_time_series is None:
            self.dm_time_series = self.get_density_matrix(self.init_state)

        self.measure_outcomes = self._get_measurement_outcomes(measure_scheme=measure_scheme)
        expectation_series_classical_shadow = {
            f'DM_{i}': {  # {'XXX-[0, 1, 2]': _value, 'XXX-[2, 3, 0]': _value, ...}
                f'{obs}-{pos}': value
                for obs, pos, value in
                zip(self.model.observables_cs['observables'], self.model.observables_cs['positions'],
                    ClassicalShadow(  # Expectation Values
                        system_size=self.q_number,
                        observables_cs=self.observables_cs
                    ).calculate_expectation(
                        measure_outcomes=self.measure_outcomes[f'DM_{i}']
                    ))
            }
            for i in tqdm(range(len(self.measure_outcomes)))
        }
        self.expectation_series_classical_shadow = expectation_series_classical_shadow
        return expectation_series_classical_shadow
