//
// Created by Weiguo Ma on 2024/7/12.
//  Original work is relating to Huang.
//  https://github.com/hsinyuan-huang/predicting-quantum-properties
//

#include <stdio.h>
#include <cmath>
#include <vector>
#include <sys/time.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <utility>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

class ClassicalShadow_backend {
private:
    // systemSize
    int systemSize;

    // Observables
    vector<vector<pair<int, int>>> observables; // [[(obsPosition, obsEncoding), (...)]]
    // \DownArrow outer[] -- systemSize; middle[] -- 3 (X, Y, Z); inner[] -- observableCounter;
    vector<vector<vector<int>>> observable_on_iQubit;

    // Measurement
    vector<vector<int>> measurementObservables;
    vector<vector<int>> measurementOutcomes;

    // Subsystem
    vector<vector<int>> subSystems;

    // Expectations of Observables
    vector<double> expectations;

    // Entropy of subsystems
    vector<double> entropies;

    //
    double renyi_sum_of_binary_outcome[10000000];
    double renyi_number_of_outcomes[10000000];

    void loadObservables(const string& fileName);
    void loadObservables4CS(const py::dict& observables4CS);

    void loadMeasurementOutcome(const string& fileName);
    void loadMeasurementOutcome_data(const py::dict& measureOutcomes);

    void loadSubsystem(const string& fileName);

public:
    ClassicalShadow_backend(int systemSize,
                    const optional<py::dict>& observables4CS,
                    const optional<string>& obsFileName) : systemSize(systemSize) {
        if (observables4CS) {
            loadObservables4CS(*observables4CS);
        } else if (obsFileName) {
            loadObservables(*obsFileName);
        } else {
            throw invalid_argument("Either observables4CS or obsFileName must be provided.");
        }
    }

    int observableNumber;

    vector<double> calculateExpectations(const optional<py::dict>& measureOutcomes,
                                         const optional<string>& measureOutcomeFileName);

    vector<double> calculateEntropy(const optional<py::dict>& subsystems,
                                    const optional<string>& subsystemFileName);
};

void ClassicalShadow_backend::loadObservables4CS(const py::dict& observables4CS) {
    systemSize = py::cast<int>(observables4CS["system_size"]);
    vector<int> _k_local = py::cast<vector<int>>(observables4CS["k_local"]);
    vector<std::string> _observables = py::cast<vector<std::string>>(observables4CS["observables"]);
    vector<vector<int>> _positions = py::cast<vector<vector<int>>>(observables4CS["positions"]);

    observableNumber = 0;
    observable_on_iQubit.assign(systemSize, vector<vector<int>>(3));

    int observableCounter = 0;

    for (size_t i = 0; i < _observables.size(); ++i) {
        int k_local = _k_local[i];
        std::string _obs = _observables[i];
        vector<int> _pos = _positions[i];

        vector<pair<int, int>> ith_observable;

        for (int k = 0; k < k_local; ++k) {
            assert(_obs[k] == 'X' || _obs[k] == 'Y' || _obs[k] == 'Z');
            int _obs_encoding = _obs[k] - 'X';

            observable_on_iQubit[_pos[k]][_obs_encoding].push_back(observableCounter);
            ith_observable.emplace_back(_pos[k], _obs_encoding);
        }
        observables.push_back(ith_observable);
        ++observableCounter;
    }
    observableNumber = observableCounter;
}

void ClassicalShadow_backend::loadObservables(const string& fileName) {
    ifstream obsFile(fileName);
    if (obsFile.fail()) {
        cerr << "\n====\nError: the input file \"%s\" does not exist.\n====\n" << fileName << endl;
        exit(-1);
    }

    int obsSystemSize;
    obsFile >> obsSystemSize;

    if (obsSystemSize != systemSize) {
        cerr << "\n====\nError: System size in observable file does not match.\n====\n" << endl;
        exit(-1);
    }

    observableNumber = 0;
    observable_on_iQubit.assign(systemSize, vector<vector<int>>(3));

    string line;
    int observableCounter = 0;

    while (getline(obsFile, line)) {
        if (line.empty()) continue;
        istringstream singleLine_stream(line);

        int k_local;
        singleLine_stream >> k_local;

        vector<pair<int, int>> ith_observable;
        for (int k = 0; k < k_local; k++) {
            char observable[5];
            int obsPosition;
            singleLine_stream >> observable >> obsPosition;

            assert(observable[0] == 'X' || observable[0] == 'Y' || observable[0] == 'Z');

            int obsEncoding = observable[0] - 'X';

            observable_on_iQubit[obsPosition][obsEncoding].push_back(observableCounter);

            ith_observable.emplace_back(obsPosition, obsEncoding);
        }
        observables.push_back(ith_observable);
        ++observableCounter;
    }
    observableNumber = observableCounter;
    obsFile.close();
}

void ClassicalShadow_backend::loadMeasurementOutcome_data(const py::dict& measureOutcomes) {
    vector<vector<string>> _orientations = py::cast<vector<vector<string>>>(measureOutcomes["orientations"]);
    vector<vector<int> >_outcomes = py::cast<vector<vector<int>>>(measureOutcomes["outcomes"]);

    for (size_t line = 0; line < _orientations.size(); ++line) {
        vector<int> orientationList(systemSize);
        vector<int> outcomeList(systemSize);

        for (int ith_qubit = 0; ith_qubit < systemSize; ++ith_qubit) {
            char obs[5];
            int binaryOutcome;

            obs[0] = _orientations[line][ith_qubit][0];
            binaryOutcome = _outcomes[line][ith_qubit];

            assert(binaryOutcome == 1 || binaryOutcome == -1);

            orientationList[ith_qubit] = obs[0] - 'X';
            outcomeList[ith_qubit] = binaryOutcome;
        }
        measurementObservables.push_back(orientationList);
        measurementOutcomes.push_back(outcomeList);
    }
}

void ClassicalShadow_backend::loadMeasurementOutcome(const string& fileName) {
    ifstream measFile(fileName);
    if (measFile.fail()) {
        cerr << "\n====\nError: the input file \"%s\" does not exist.\n====\n" << fileName << endl;
        exit(-1);
    }

    int measSystemSize;
    measFile >> measSystemSize;
    if (measSystemSize != systemSize) {
        cerr << "\n====\nError: System size in observable file does not match.\n====\n" << endl;
        exit(-1);
    }

    string line;
    while (getline(measFile, line)) {
        if (line.empty()) continue;
        istringstream singleLine_stream(line);

        vector<int> observableList(systemSize);
        vector<int> outcomeList(systemSize);

        for (int ith_qubit = 0; ith_qubit < systemSize; ++ith_qubit) {
            char observable[5];
            int binaryOutcome;

            singleLine_stream >> observable >> binaryOutcome;

            assert(binaryOutcome == 1 || binaryOutcome == -1);

            observableList[ith_qubit] = observable[0] - 'X';
            outcomeList[ith_qubit] = binaryOutcome;
        }
        measurementObservables.push_back(observableList);
        measurementOutcomes.push_back(outcomeList);
    }
    measFile.close();
}

void ClassicalShadow_backend::loadSubsystem(const string& fileName) {
    ifstream subSystemFile(fileName);
    if (subSystemFile.fail()) {
        cerr << "\n====\nError: the input file \"%s\" does not exist.\n====\n" << fileName << endl;
        exit(-1);
    }

    int subsystemSize;
    subSystemFile >> subsystemSize;
    if (subsystemSize > systemSize) {
        cerr << "\n====\nError: System size in Subsystem file is not equal to the system size.\n====\n" << endl;
        exit(-1);
    }

    string line;
    while (getline(subSystemFile, line)) {
        if (line.empty()) continue;
        istringstream singleLine_stream(line);

        int k_local;
        singleLine_stream >> k_local;

        vector<int> ith_subsystem;
        for (int k = 0; k < k_local; ++k) {
            int qubitPosition;
            singleLine_stream >> qubitPosition;
            ith_subsystem.push_back(qubitPosition);
        }
        subSystems.push_back(ith_subsystem);
    }
    subSystemFile.close();
}

vector<double> ClassicalShadow_backend::calculateExpectations(const optional<py::dict>& measureOutcomes,
                                                              const optional<string>& measureOutcomeFileName) {
    vector<int> obsMatchCount(observableNumber);
    vector<int> cumulativeMeasurements(observableNumber);
    vector<int> measurementNumber(observableNumber);
    vector<int> measurementResultSum(observableNumber);

    if (measureOutcomes) {
        loadMeasurementOutcome_data(*measureOutcomes);
    } else if (measureOutcomeFileName) {
        loadMeasurementOutcome(*measureOutcomeFileName);
    } else {
        throw invalid_argument("Either measureOutcomes or measureOutcomeFileName must be provided.");
    }

    for (int measEpoch = 0;
         measEpoch < (int) measurementObservables.size(); measEpoch++) { // number of measurement scheme
        for (int i = 0; i < (int) observables.size(); i++) { // number of observables
            obsMatchCount[i] = observables[i].size(); // k-local of observables[i]
            cumulativeMeasurements[i] = 1;
        }
        for (int ith_qubit = 0; ith_qubit < systemSize; ith_qubit++) {
            int observable = measurementObservables[measEpoch][ith_qubit];
            int outcome = measurementOutcomes[measEpoch][ith_qubit];

            for (int i: observable_on_iQubit[ith_qubit][observable]) { // for element in 0-th qubit X/Y/Z -- [order of obs]
                obsMatchCount[i]--; // k-local number
                cumulativeMeasurements[i] *= outcome; // e.g., [-1, 1] for 2-local observable
            }
        }

        for (int i = 0; i < (int) observables.size(); i++) { // number of observables
            if (obsMatchCount[i] == 0) {
                measurementNumber[i]++;
                measurementResultSum[i] += cumulativeMeasurements[i];
            }
        }
    }

    for (int i = 0; i < (int) observables.size(); i++) {
        if (measurementNumber[i] == 0) {
            expectations.push_back(0);
        } else {
            expectations.push_back((double) measurementResultSum[i] / measurementNumber[i]);
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
            .def(py::init<int, const optional<py::dict>&, const optional<string>&>(),
                    py::arg("systemSize"),
                    py::arg("observables4CS") = py::none(),
                    py::arg("obsFileName") = py::none())
            .def("calculateExpectations", &ClassicalShadow_backend::calculateExpectations,
                 py::arg("measureOutcomes") = py::none(),
                 py::arg("measureOutcomeFileName") = py::none())
            .def("calculateEntropy", &ClassicalShadow_backend::calculateEntropy,
                    py::arg("subsystems") = py::none(),
                    py::arg("subsystemFileName") = py::none())
            .def_readonly("observableNumber", &ClassicalShadow_backend::observableNumber);
}