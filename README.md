# ShadowPyExpect

```markdown
# Z2 Model Expectation Calculation

This repository demonstrates the use of the Z2 Model for calculating quantum expectations. The project includes Python scripts, C++ backend integration, and example files for testing and demonstration.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Make sure you have the following software installed:

- Python 3.10+
- NumPy
- QuTiP

Install the necessary Python packages using `pip`:

```bash
pip install numpy qutip
```

### Project Structure

```
.
├── CMakeLists.txt                  # CMake configuration
├── ExampleFiles/                   # Example data files
├── TestFiles/                      # Test data files
├── Z2Measurement/                  # Main measurement module
├── build/                          # Build directory for C++ components
├── cmake-build-debug/              # Debug build directory
├── extern/                         # External dependencies
├── src/
│   └── python/
│       ├── __init__.py             # Initialization file for the Python module
│       ├── abstract_properties.py  # Abstract properties module
│       ├── cpp_to_python_interface.py  # C++ to Python interface
│       ├── fake_sampler.py         # Fake sampler module
│       ├── generate_fake_measurement.py # Generate fake measurement module
│       ├── observables_generator.py # Observables generator module
│       ├── z2_model.py             # Z2 model implementation
├── main.py                         # Main script for running the demonstration
├── README.md                       # Project documentation
```

### Running the Program

To run the demonstration program, execute the following command:

```bash
python main.py
```

### Code Overview

The `main.py` script demonstrates the use of the `Z2Model` class for calculating quantum expectations. Here's a brief overview of its main components:

#### `main()`

- Initializes the density matrix for a 4-qubit system.
- Creates an instance of `Z2Model`.
- Calculates expectations using both quantum simulation and classical simulation.
- Analyzes and prints the results.

#### `initialize_density_matrix()`

- Initializes the density matrix for a 4-qubit system using tensor products of basis states.
- Converts the state to a density matrix.

#### `analyze_results(expectations_cs)`

- Analyzes the results of the expectation calculations, including magnetization and Gauss law expectations.

#### `print_results(magnetization, gauss0, gauss1)`

- Prints the results of the magnetization and Gauss law expectations, including mean values and standard deviations.

### Example Output

After running the program, you should see an output similar to this:

```
Conservation Magnetization Hamiltonian - Expectation:
[1.0, 1.0, 1.0, ...]

Mean Value: 0.0 and the Standard Deviation: 0.0, length: 300

Conservation Gauss Law Hamiltonian - Expectation:
Expectation of Gauss Law Hamiltonian 1:
[0.0, 0.0, 0.0, ...]

Mean Value: 1.0 and the Standard Deviation: 0.0, length: 300

Expectation of Gauss Law Hamiltonian 2:
[0.0, 0.0, 0.0, ...]

Mean Value: 1.0 and the Standard Deviation: 0.0, length: 300
```

## Contributing

To contribute to this project, fork the repository and submit a pull request.

## Acknowledgments

- [QuTiP](http://qutip.org/) - Quantum Toolbox in Python
- [NumPy](https://numpy.org/) - Fundamental package for scientific computing with Python