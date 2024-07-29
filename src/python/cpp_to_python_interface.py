import os
import sys
from typing import Optional, List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build')))

from expectationCS import ClassicalShadow_backend
from generateMeasureScheme import MeasureScheme_backend


class ClassicalShadow:
    def __init__(self,
                 system_size: int,
                 observables_cs: Optional[Dict] = None,
                 obs_file_name: Optional[str] = None):
        """
        Python interface for the classical shadow cpp-backend.

        Args:
            system_size (int): The size of the system.
            observables_cs (Optional[Dict]): Observables for Classical Shadow.
            obs_file_name (Optional[str]): The file name of the observables.
        """
        self._classical_shadow = ClassicalShadow_backend(system_size=system_size,
                                                         observables_cs=observables_cs)

    def calculate_expectation(self, measure_outcomes: Optional[Dict] = None) -> List[float]:
        """
        Calculate the expectation values using the measurement outcome file.

        Args:
            measure_outcomes (Optional[Dict]): The measurement outcomes.
            measure_outcome_file_name (Optional[str]): The file name of the measurement outcomes.

        Returns:
            List[float]: A list of expectation values.

        Raises:
            ValueError: If neither measure_outcomes nor measure_outcome_file_name is provided.
        """
        if measure_outcomes is not None:
            expectations = self._classical_shadow.calculateExpectations(measureOutcomes=measure_outcomes)
        else:
            raise ValueError('You must supply a series of measurement outcomes for calculating the expectations.')

        return expectations

    def calculate_entropies(self, subsystem_file_name: str) -> List[float]:
        """
        Calculate the entropies using the subsystem file.

        Args:
            subsystem_file_name (str): The file name of the subsystem.

        Returns:
            List[float]: A list of entropy values.
        """
        entropies = self._classical_shadow.calculateEntropy(subsystem_file_name)
        return entropies


class MeasureScheme:
    def __init__(self, system_size: int):
        """
        Python interface for the measurement scheme cpp-backend.
        """
        self._measure_scheme = MeasureScheme_backend(systemSize=system_size)
        self.eta = self._measure_scheme.eta

        self.random_scheme, self.derandom_scheme = None, None

    def random_generate(self, total_measurement_times: int, output_file_name: Optional[str] = None) -> List[str]:
        """
        Generate a random measurement scheme.

        Args:
            total_measurement_times (int): Total number of measurement times.
            output_file_name (Optional[str]): File name for the output scheme.

        Returns:
            List[str]: The generated measurement scheme.
        """
        self.random_scheme = self._measure_scheme.randomGenerate(
            totalMeasurementTimes=total_measurement_times
        )
        return self.random_scheme

    def derandom_generate(self, measurement_times_per_observable: int,
                          observables_cs: Optional[Dict] = None) -> List[str]:
        """
        Generate a de-randomized measurement scheme.

        Args:
            measurement_times_per_observable (int): Number of measurement times per observable.
            observables_cs (Optional[Dict]): Observables for Classical Shadow.

        Returns:
            List[str]: The generated de-randomized measurement scheme.

        Raises:
            ValueError: If neither observables_cs nor observables_file_name is provided.
        """
        if observables_cs is not None:
            self.derandom_scheme = self._measure_scheme.deRandomGenerate(
                measurementTimes_perObservable=measurement_times_per_observable,
                observables_cs=observables_cs
            )
        else:
            raise ValueError('You must supply a series of observables for the de-random scheme.')

        return self.derandom_scheme
