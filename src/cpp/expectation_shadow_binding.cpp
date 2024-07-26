//
// Created by Weiguo Ma on 2024/7/12.
//  Original work is relating to Huang.
//  https://github.com/hsinyuan-huang/predicting-quantum-properties
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

class ClassicalShadow_backend {
private:
    int systemSize;
    vector<vector<pair<int, int>>> observables;
    vector<vector<vector<int>>> observable_on_iQubit;
    vector<vector<int>> measurementObservables;
    vector<vector<int>> measurementOutcomes;
    vector<vector<int>> subSystems;
    vector<double> expectations;
    vector<double> entropies;
    double renyi_sum_of_binary_outcome[10000000] = {0};
    double renyi_number_of_outcomes[10000000] = {0};

    void loadObservables_cs(const py::dict& observables4CS);
    void loadMeasurementOutcome(const py::dict& measureOutcomes);
    void loadSubsystem(const string& fileName);

public:
    ClassicalShadow_backend(int systemSize, const optional<py::dict>& observables4CS);

    int observableNumber;
    vector<double> calculateExpectations(const py::dict& measureOutcomes);
    vector<double> calculateEntropy(const optional<py::dict>& subsystems, const optional<string>& subsystemFileName);
};

ClassicalShadow_backend::ClassicalShadow_backend(int systemSize, const optional<py::dict>& observables4CS) : systemSize(systemSize) {
    if (observables4CS) {
        loadObservables_cs(*observables4CS);
    } else {
        throw invalid_argument("Observables4CS must be provided.");
    }
}

void ClassicalShadow_backend::loadObservables_cs(const py::dict& observables4CS) {
    try {
        systemSize = py::cast<int>(observables4CS["system_size"]);
        vector<int> kLocal = py::cast<vector<int>>(observables4CS["k_local"]);
        vector<string> observablesList = py::cast<vector<string>>(observables4CS["observables"]);
        vector<vector<int>> positions = py::cast<vector<vector<int>>>(observables4CS["positions"]);

        observableNumber = 0;
        observable_on_iQubit.assign(systemSize, vector<vector<int>>(3));

        int observableCounter = 0;

        for (size_t i = 0; i < observablesList.size(); ++i) {
            int kLocalValue = kLocal[i];
            string observable = observablesList[i];
            vector<int> position = positions[i];

            vector<pair<int, int>> ithObservable;

            for (int k = 0; k < kLocalValue; ++k) {
                if (observable[k] != 'X' && observable[k] != 'Y' && observable[k] != 'Z') {
                    throw invalid_argument("Invalid observable character. Must be X, Y, or Z.");
                }
                int obsEncoding = observable[k] - 'X';

                observable_on_iQubit[position[k]][obsEncoding].push_back(observableCounter);
                ithObservable.emplace_back(position[k], obsEncoding);
            }
            observables.push_back(ithObservable);
            ++observableCounter;
        }
        observableNumber = observableCounter;
    } catch (const exception& e) {
        cerr << "Error loading observables: " << e.what() << endl;
        throw;
    }
}

void ClassicalShadow_backend::loadMeasurementOutcome(const py::dict& measureOutcomes) {
    try {
        vector<vector<string>> orientations = py::cast<vector<vector<string>>>(measureOutcomes["orientations"]);
        vector<vector<int>> outcomes = py::cast<vector<vector<int>>>(measureOutcomes["outcomes"]);

        for (size_t line = 0; line < orientations.size(); ++line) {
            vector<int> orientationList(systemSize);
            vector<int> outcomeList(systemSize);

            for (int ithQubit = 0; ithQubit < systemSize; ++ithQubit) {
                char orientationChar = orientations[line][ithQubit][0];
                int binaryOutcome = outcomes[line][ithQubit];

                if (binaryOutcome != 1 && binaryOutcome != -1) {
                    throw invalid_argument("Invalid measurement outcome. Must be 1 or -1.");
                }

                orientationList[ithQubit] = orientationChar - 'X';
                outcomeList[ithQubit] = binaryOutcome;
            }
            measurementObservables.push_back(orientationList);
            measurementOutcomes.push_back(outcomeList);
        }
    } catch (const exception& e) {
        cerr << "Error loading measurement outcomes: " << e.what() << endl;
        throw;
    }
}

void ClassicalShadow_backend::loadSubsystem(const string& fileName) {
    ifstream subSystemFile(fileName);
    if (!subSystemFile) {
        throw runtime_error("Error: the input file \"" + fileName + "\" does not exist.");
    }

    int subsystemSize;
    subSystemFile >> subsystemSize;
    if (subsystemSize > systemSize) {
        throw runtime_error("Error: System size in Subsystem file exceeds the system size.");
    }

    string line;
    while (getline(subSystemFile, line)) {
        if (line.empty()) continue;
        istringstream lineStream(line);

        int kLocal;
        lineStream >> kLocal;

        vector<int> subsystem;
        for (int k = 0; k < kLocal; ++k) {
            int qubitPosition;
            lineStream >> qubitPosition;
            subsystem.push_back(qubitPosition);
        }
        subSystems.push_back(subsystem);
    }
    subSystemFile.close();
}

vector<double> ClassicalShadow_backend::calculateExpectations(const py::dict& measureOutcomes) {
    vector<int> obsMatchCount(observableNumber);
    vector<int> cumulativeMeasurements(observableNumber);
    vector<int> measurementNumber(observableNumber, 0);
    vector<int> measurementResultSum(observableNumber, 0);

    loadMeasurementOutcome(measureOutcomes);

    for (size_t measEpoch = 0; measEpoch < measurementObservables.size(); ++measEpoch) {
        for (size_t i = 0; i < observables.size(); ++i) {
            obsMatchCount[i] = observables[i].size();
            cumulativeMeasurements[i] = 1;
        }

        for (int ithQubit = 0; ithQubit < systemSize; ++ithQubit) {
            int observable = measurementObservables[measEpoch][ithQubit];
            int outcome = measurementOutcomes[measEpoch][ithQubit];

            for (int obsIdx : observable_on_iQubit[ithQubit][observable]) {
                obsMatchCount[obsIdx]--;
                cumulativeMeasurements[obsIdx] *= outcome;
            }
        }

        for (size_t i = 0; i < observables.size(); ++i) {
            if (obsMatchCount[i] == 0) {
                measurementNumber[i]++;
                measurementResultSum[i] += cumulativeMeasurements[i];
            }
        }
    }

    expectations.clear();
    expectations.reserve(observables.size());
    for (size_t i = 0; i < observables.size(); ++i) {
        if (measurementNumber[i] == 0) {
            expectations.push_back(0.0);
        } else {
            expectations.push_back(static_cast<double>(measurementResultSum[i]) / measurementNumber[i]);
        }
    }
    return expectations;
}


vector<double> ClassicalShadow_backend::calculateEntropy(const optional<py::dict>& subsystems,
                                                         const optional<string>& subsystemFileName) {
    if (subsystemFileName) {
        loadSubsystem(*subsystemFileName);
    } else if (subsystems) {
        throw invalid_argument("Not Implemented");
    } else {
        throw invalid_argument("You have to provide either subsystems in dict or subsystemFileName.");
    }

    for (const auto &subSystem: subSystems) {
        int subsystemSize = subSystem.size();
        fill(renyi_sum_of_binary_outcome, renyi_sum_of_binary_outcome + (1 << (2 * subsystemSize)), 0);
        fill(renyi_number_of_outcomes, renyi_number_of_outcomes + (1 << (2 * subsystemSize)), 0);

        for (int t = 0; t < measurementObservables.size(); ++t) {
            long long encoding = 0, cumulative_outcome = 1;
            renyi_sum_of_binary_outcome[0] += 1;
            renyi_number_of_outcomes[0] += 1;

            for (long long b = 1; b < (1 << subsystemSize); ++b) {
                long long change_i = __builtin_ctzll(b);
                long long index_in_original_system = subSystem[change_i];
                cumulative_outcome *= measurementOutcomes[t][index_in_original_system];
                encoding ^= (measurementObservables[t][index_in_original_system] + 1) << (2LL * change_i);
                renyi_sum_of_binary_outcome[encoding] += cumulative_outcome;
                renyi_number_of_outcomes[encoding] += 1;
            }
        }

        vector<int> level_cnt(subsystemSize + 1, 0);
        vector<int> level_ttl(subsystemSize + 1, 0);

        for (long long c = 0; c < (1 << (2 * subsystemSize)); ++c) {
            int nonId = 0;
            for (int i = 0; i < subsystemSize; ++i) {
                nonId += ((c >> (2 * i)) & 3) != 0;
            }
            if (renyi_number_of_outcomes[c] >= 2) level_cnt[nonId]++;
            level_ttl[nonId]++;
        }

        double predicted_entropy = 0;
        for (long long c = 0; c < (1 << (2 * subsystemSize)); ++c) {
            if (renyi_number_of_outcomes[c] <= 1) continue;
            int nonId = 0;
            for (int i = 0; i < subsystemSize; ++i)
                nonId += ((c >> (2 * i)) & 3) != 0;

            predicted_entropy += (1.0 / (renyi_number_of_outcomes[c] * (renyi_number_of_outcomes[c] - 1))) *
                                 (renyi_sum_of_binary_outcome[c] * renyi_sum_of_binary_outcome[c] -
                                  renyi_number_of_outcomes[c]) /
                                 (1LL << subsystemSize) * level_ttl[nonId] / level_cnt[nonId];
        }
        entropies.push_back(
                (double) -1.0 * log2(min(max(predicted_entropy, 1.0 / pow(2.0, subsystemSize)), 1.0 - 1e-9)));
    }
    return entropies;
}

PYBIND11_MODULE(expectationCS, m) {
    py::class_<ClassicalShadow_backend>(m, "ClassicalShadow_backend")
            .def(py::init<int, const optional<py::dict>&>(),
                    py::arg("system_size"),
                    py::arg("observables_cs") = py::none())
            .def("calculateExpectations", &ClassicalShadow_backend::calculateExpectations,
                 py::arg("measureOutcomes"))
            .def("calculateEntropy", &ClassicalShadow_backend::calculateEntropy,
                    py::arg("subsystems") = py::none(),
                    py::arg("subsystemFileName") = py::none())
            .def_readonly("observableNumber", &ClassicalShadow_backend::observableNumber);
}