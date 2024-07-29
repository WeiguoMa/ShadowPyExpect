from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from numpy import ndarray
from tqdm import tqdm

from .cpp_to_python_interface import MeasureScheme, ClassicalShadow
from .fake_sampler import FakeSampler


class AbstractModelProperties(ABC):
    def __init__(self,
                 system_size: int,
                 observables_cs: Optional[Dict] = None):
        """
        Abstract class for initializing model properties.

        Args:
            system_size (int): The size of the system.
            observables_cs (Optional[Dict]): Observables for Classical Shadow.
                Note that the observables for the Classical Shadow should be a constant variable,
                while the density matrix series can be modified in .calculate_expectation().
        """
        self.q_number = system_size

        self.init_state = None
        self.dm_time_series = None
        self.observables_cs = observables_cs if observables_cs is not None else None

        # tools
        self.measure_scheme_generator = MeasureScheme(system_size=system_size)
        self.fake_sampler = FakeSampler(system_size=system_size)

        self.measure_scheme, self.measure_outcomes = None, None
        self.expectation_series_classical_shadow = None

    @abstractmethod
    def calculate_expectation(self, measurement_times: int,
                              derandom_scheme: bool = True,
                              density_matrix_series: Optional[List] = None):
        """
        Calculates the expectation values using ShadowPyExpect.
        """
        self.dm_time_series = density_matrix_series

        return self._calculate_expectation_classical_shadow(
            density_matrix_series=density_matrix_series,
            measurement_times=measurement_times,
            derandom_scheme=derandom_scheme
        )

    def _calculate_expectation_classical_shadow(self,
                                                density_matrix_series: List[ndarray],
                                                measurement_times: int,
                                                derandom_scheme: bool = True):
        """
        Args:
            density_matrix_series (List[ndarray]): The series of density matrices.
            measurement_times (int): The number of measurement times.
            derandom_scheme (bool): Flag to derandomize the measurement scheme.
        """
        self.measure_scheme = self._generate_measure_scheme(
            measurement_times=measurement_times, derandom_scheme=derandom_scheme
        )

        self.measure_outcomes = self._get_measurement_outcomes(
            density_matrix_series=density_matrix_series,
            measure_scheme=self.measure_scheme
        )

        expectation_series_classical_shadow = {
            f'DM_{i}': {  # {'XXX-[0, 1, 2]': _value, 'XXX-[2, 3, 0]': _value, ...}
                f'{obs}-{pos}': value
                for obs, pos, value in
                zip(self.observables_cs['observables'], self.observables_cs['positions'],
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
                total_measurement_times=measurement_times
            )

        return measure_scheme

    def _get_measurement_outcomes(self, density_matrix_series: List[ndarray],
                                  measure_scheme: Optional[List] = None) -> Dict[str, Dict[str, List]]:
        if measure_scheme is None:
            raise ValueError("The measurement scheme must be provided.")

        return {
            f'DM_{i}': {  # ori: ['X', 'X', 'X', 'X'] -> outcome: [1, -1, -1, 1]
                'orientations': list(orientations),
                'outcomes': [[int(outcome) for outcome in sublist] for sublist in outcomes]  # str -> int
            }
            for i, _dm in tqdm(enumerate(density_matrix_series))
            for orientations, outcomes in
            [zip(*[self.fake_sampler.fake_sampling_dm(_dm, scheme) for scheme in measure_scheme])]
        }
