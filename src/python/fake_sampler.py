import os
import sys
from itertools import product
from typing import Union, List, Optional

import numpy as np
from qutip import Qobj

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build')))


class FakeSampler:
    def __init__(self, system_size: int, basis: Optional[List[str]] = None, circuit: bool = False):
        """
        Initialize the FakeSampler.

        Args:
            system_size (int): The size of the system.
            basis (Optional[List[str]]): The basis states.
            circuit (bool): Flag to indicate if a circuit is used.
        """
        self.basis = [''.join(state) for state in product('01', repeat=system_size)] if basis is None else basis
        from fakeSampler_backend import FakeSampler_backend
        self.backend = FakeSampler_backend(proj_basis=self.basis)

    def fake_sampling_dm(self, dm: Union[np.ndarray, Qobj],
                         measurement_orientation: Optional[List[str]] = None):
        """
        Perform fake sampling on a density matrix.

        Args:
            dm (Union[np.ndarray, Qobj]): The density matrix.
            measurement_orientation (Optional[List[str]]): The measurement orientations.

        Returns:
            Tuple[List[str], List[int]]: The measurement orientations and their corresponding eigenvalues.

        Raises:
            TypeError: If the density matrix is not of type np.ndarray or qutip.Qobj.
            ValueError: If the length of measurement_orientation does not match the system size.
        """
        system_size = int(np.log2(dm.shape[0]))

        if isinstance(dm, Qobj):
            dm = dm.full()
        elif not isinstance(dm, np.ndarray):
            raise TypeError("The type of Density Matrix must be np.ndarray or qutip.Qobj.")

        if measurement_orientation is None:
            measurement_orientation = ['Z'] * system_size
        elif len(measurement_orientation) != system_size:
            raise ValueError("The length of measurement_orientation must be equal to the system size.")

        state_eigenvalues = self.backend.fakeSampling_dm(dm_array=dm, measurement_orientation=measurement_orientation)

        return measurement_orientation, state_eigenvalues
