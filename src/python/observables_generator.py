import warnings
from typing import List, Optional, Tuple


def generate_observables(system_size: int,
                         repeat_item: List[str],
                         repeat_period: List[List[int]],
                         write: bool = False,
                         write_file_name: Optional[str] = None,
                         write_mode: str = 'w') -> Tuple[List[str], List[List[int]]]:
    """
    Generate observables for a quantum system.

    Args:
        system_size (int): Number of qubits.
        repeat_item (List[str]): List of items to repeat in one k-local observable.
        repeat_period (List[List[int]]): List of periods for repetition.
        write (bool): Whether to write to a file.
        write_file_name (Optional[str]): Name of the file to write to.
        write_mode (str): Mode to write the file in.

    Returns:
        Tuple[List[str], List[List[int]]]: Observables and their positions.

    Example:
        >>> generate_observables(4, ['+', 'X', '-'], [[0, 1, 2], [1, 2, 3]])
        >>> (['XXX', 'XXY', 'YXX', 'YXY'], [[0, 1, 2], [1, 2, 3]])
    """

    def transpile_obs(string_list: List[str]) -> List[str]:
        if not string_list:
            return ['']

        replacements = {
            '+': ['X', 'Y'],
            '-': ['X', 'Y'],
            '*': ['X', 'Y'],
            '/': ['X', 'Y'],
        }

        first_char = string_list[0]
        rest_expanded = transpile_obs(string_list[1:])

        if first_char in replacements:
            expanded = []
            for replacement in replacements[first_char]:
                for rest in rest_expanded:
                    expanded.append(replacement + rest)
            return expanded
        else:
            return [first_char + rest for rest in rest_expanded]

    k_local = len(repeat_item)
    observable = transpile_obs(repeat_item)

    period = repeat_period
    for i, ele in enumerate(period):
        if ele[0] == -1:
            period[i][0] = system_size - 1
        if ele[-1] == system_size:
            period[i][-1] = 0

    observables = [ele for ele in observable for _ in range(len(repeat_period))]
    positions = period * len(observable)

    if write:
        if write_file_name is None:
            warnings.warn("The file name should be provided. Using default file name './observablesTemp.txt'.")
            write_file_name = './observablesTemp.txt'
        zip_list = [zip(obs, pos) for obs in observable for pos in period]
        with open(write_file_name, write_mode) as write_file:
            if write_mode == 'w':
                write_file.write(f"{system_size}\n")
            for element in zip_list:
                string = ' '.join([' '.join([obs, str(pos)]) for obs, pos in element])
                write_file.write(f'{k_local} ' + f"{string}\n")

    return observables, positions
