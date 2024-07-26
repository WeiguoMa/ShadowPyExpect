//
// Created by Weiguo Ma on 2024/7/11.
//  Original work is relating to Huang.
//  https://github.com/hsinyuan-huang/predicting-quantum-properties
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <cassert>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

const int INF = numeric_limits<int>::max();

class MeasureScheme_backend {
private:
    int sum_cnt;
    int systemSize;
    double sum_logValue;
    int observableNumber;

    vector<double> observables_weight;
    vector<vector<vector<int>>> observables_on_iQubit;

    vector<vector<char>> randomScheme;
    vector<vector<char>> deRandomScheme;

    vector<double> log1ppow1o3k;

    void loadObservables(const string& fileName);
    void loadObservablesData(const py::dict& observables4CS);

    double fail_probPessimistic(int measurementTimes_perObservable, int curMeasurementTimes, int obsMatchCount,
                                double weight, double shift);

public:
    MeasureScheme_backend() : eta(0.9),
                              systemSize(0),
                              observableNumber(0),
                              max_k_local(0),
                              sum_cnt(0),
                              sum_logValue(0) {}  // Default Constructor
    double eta;
    int max_k_local;

    vector<vector<pair<int, int>>> observables;

    vector<vector<char>> randomGenerate(int totalMeasurementTimes,
                                        optional<string> outputFileName = nullopt);

    vector<vector<char>> deRandomGenerate(int measurementTimes_perObservable,
                                          const optional<py::dict>& observables4CS,
                                          const optional<string>& observablesFileName,
                                          const optional<string>& outputFileName);
};

void MeasureScheme_backend::loadObservablesData(const py::dict& observables4CS) {
    try {
        systemSize = py::cast<int>(observables4CS["system_size"]);
        vector<int> _k_local = py::cast<vector<int>>(observables4CS["k_local"]);
        vector<std::string> _observables = py::cast<vector<std::string>>(observables4CS["observables"]);
        vector<vector<int>> _positions = py::cast<vector<vector<int>>>(observables4CS["positions"]);

        vector<double> _weights(_observables.size(), 1.0);
        if (observables4CS.contains("weights")) {
            _weights = py::cast<vector<double>>(observables4CS["weights"]);
            if (_weights.size() != _observables.size()) {
                throw std::runtime_error("Size of weights does not match size of observables.");
            }
        }

        max_k_local = 0;
        observableNumber = 0;
        observables_on_iQubit.clear();

        vector<int> positions;
        vector<vector<int>> obsList(3, positions);

        observables_on_iQubit.resize(systemSize, obsList);

        int observableCounter = 0;

        for (size_t i = 0; i < _observables.size(); ++i) {
            int k_local = _k_local[i];
            std::string _obs = _observables[i];
            vector<int> _pos = _positions[i];

            if (_pos.size() != static_cast<size_t>(k_local)) {
                throw std::runtime_error("Size of positions does not match k_local.");
            }

            max_k_local = std::max(max_k_local, k_local);
            vector<std::pair<int, int>> ith_observable;

            for (int k = 0; k < k_local; ++k) {
                if (_obs[k] != 'X' && _obs[k] != 'Y' && _obs[k] != 'Z') {
                    throw std::runtime_error("Invalid observable character. Must be 'X', 'Y', or 'Z'.");
                }
                int _obs_encoding = _obs[k] - 'X';

                observables_on_iQubit[_pos[k]][_obs_encoding].push_back(observableCounter);
                ith_observable.emplace_back(_pos[k], _obs_encoding);
            }
            observables_weight.push_back(_weights[i]);
            observables.push_back(ith_observable);

            ++observableCounter;
        }
        observableNumber = observableCounter;
    } catch (const py::cast_error& e) {
        throw std::runtime_error("Type casting error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error in loadObservablesData: " + std::string(e.what()));
    }
}


void MeasureScheme_backend::loadObservables(const string &fileName) {
    ifstream observableFile(fileName);
    if (!observableFile) {
        cerr << "Error: The input file \"" << fileName << "\" does not exist." << endl;
        exit(EXIT_FAILURE);
    }

    cerr << "File Opened Successfully: \"" << fileName << "\"" << endl;
    observableFile >> systemSize;

    max_k_local = 0;
    observableNumber = 0;
    observables_on_iQubit.clear();

    vector<int> positions;
    vector<vector<int>> obsList(3, positions);

    for (int i = 0; i < systemSize; ++i) {
        observables_on_iQubit.push_back(obsList);
    }

    string line;
    int observableCounter = 0;

    while (getline(observableFile, line)) {
        if (line.empty()) continue;
        istringstream singleLine_stream(line);

        int k_local;
        singleLine_stream >> k_local;
        max_k_local = max(max_k_local, k_local);

        vector<pair<int, int>> ith_observable;
        for (int k = 0; k < k_local; ++k) {
            string pauliObservable;
            int pauliPosition;

            singleLine_stream >> pauliObservable >> pauliPosition;

            assert(pauliObservable[0] == 'X' || pauliObservable[0] == 'Y' || pauliObservable[0] == 'Z');
            int pauliCoding = pauliObservable[0] - 'X'; // ASCII Coding

            observables_on_iQubit[pauliPosition][pauliCoding].push_back(observableCounter);
            ith_observable.emplace_back(pauliPosition, pauliCoding);
        }

        double weight = 1.0;
        if (singleLine_stream.rdbuf()->in_avail() != 0) {
            singleLine_stream >> weight;
        }

        observables_weight.push_back(weight);
        observables.push_back(ith_observable);

        ++observableCounter;
    }
    observableNumber = observableCounter;
    observableFile.close();
}

double MeasureScheme_backend::fail_probPessimistic(int measurementTimes_perObservable,
                                                   int curMeasurementTimes,
                                                   int obsMatchCount,
                                                   double weight,
                                                   double shift) {
    double log1pp0 = (obsMatchCount < INF ? log1ppow1o3k[obsMatchCount] : 0.0);

    if (floor(weight * measurementTimes_perObservable) <= curMeasurementTimes) {
        return 0;
    }

    double log_value = -eta / 2 * curMeasurementTimes + log1pp0;
    sum_logValue += (log_value / weight);
    sum_cnt++;

    return 2 * exp((log_value / weight) - shift);
}

vector<vector<char>> MeasureScheme_backend::randomGenerate(int totalMeasurementTimes,
                                                           optional<string> outputFileName) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 2);

    const char Obs[] = {'X', 'Y', 'Z'};

    ostream *out = &cout;
    ofstream outFile;

    if (outputFileName) {
        outFile.open(*outputFileName);
        if (!outFile.is_open()) {
            cerr << "Error: Could not open file " << *outputFileName << " for writing." << endl;
            return {};
        }
        out = &outFile;
    }

    for (int i = 0; i < totalMeasurementTimes; ++i) {
        vector<char> ith_measurement;
        for (int j = 0; j < systemSize; ++j) {
            char obs = Obs[dis(gen)];
            ith_measurement.push_back(obs);
            *out << obs << " ";
        }
        randomScheme.push_back(ith_measurement);
        *out << endl;
    }

    if (outFile.is_open()) {
        outFile.close();
    }
    return randomScheme;
}

vector<vector<char>> MeasureScheme_backend::deRandomGenerate(int measurementTimes_perObservable,
                                                             const optional<py::dict>& observables4CS,
                                                             const optional<string>& observablesFileName,
                                                             const optional<string>& outputFileName) {
    if (observables4CS) {
        loadObservablesData(*observables4CS);
    } else if (observablesFileName) {
        loadObservables(*observablesFileName);
    } else {
        throw invalid_argument("Either observablesFileName or pythonObservables must be provided.");
    }

    double expm1eta = expm1(-eta / 2);

    for (int k = 0; k < max_k_local; ++k) {
        log1ppow1o3k.push_back(log1p(pow(1.0 / 3.0, k) * expm1eta));
    }

    vector<int> cur_num_of_measurements(observableNumber, 0);
    vector<int> obsMatchCount(observableNumber);

    for (int measurement_repetition = 0;
         measurement_repetition < numeric_limits<int>::max(); ++measurement_repetition) {
        vector<char> single_measurement(systemSize);

        for (int i = 0; i < observableNumber; ++i) {
            obsMatchCount[i] = observables[i].size();
        }
        double shift = (sum_cnt == 0) ? 0 : sum_logValue / sum_cnt;
        sum_logValue = 0.0;
        sum_cnt = 0;

        for (int ith_qubit = 0; ith_qubit < systemSize; ++ith_qubit) {
            array<double, 3> prob_of_failure = {0, 0, 0};
            double smallest_prob_of_failure = numeric_limits<double>::max();

            for (int obs = 0; obs < 3; ++obs) {
                for (int p = 0; p < 3; ++p) {
                    for (int i: observables_on_iQubit[ith_qubit][p]) {
                        if (obs == p) {
                            int pauli_to_match_next_step = (obsMatchCount[i] == INF)
                                                           ? INF :
                                                           obsMatchCount[i] - 1;
                            double prob_next_step = fail_probPessimistic(measurementTimes_perObservable,
                                                                         cur_num_of_measurements[i],
                                                                         pauli_to_match_next_step,
                                                                         observables_weight[i], shift);
                            double prob_current_step = fail_probPessimistic(measurementTimes_perObservable,
                                                                            cur_num_of_measurements[i],
                                                                            obsMatchCount[i],
                                                                            observables_weight[i], shift);
                            prob_of_failure[obs] += prob_next_step - prob_current_step;
                        } else {
                            double prob_next_step = fail_probPessimistic(measurementTimes_perObservable,
                                                                         cur_num_of_measurements[i],
                                                                         numeric_limits<int>::max(),
                                                                         observables_weight[i], shift);
                            double prob_current_step = fail_probPessimistic(measurementTimes_perObservable,
                                                                            cur_num_of_measurements[i],
                                                                            obsMatchCount[i],
                                                                            observables_weight[i], shift);
                            prob_of_failure[obs] += prob_next_step - prob_current_step;
                        }
                    }
                }
                smallest_prob_of_failure = min(smallest_prob_of_failure, prob_of_failure[obs]);
            }

            int _bestOBS = distance(prob_of_failure.begin(),
                                    min_element(prob_of_failure.begin(),
                                                prob_of_failure.end()));

            single_measurement[ith_qubit] = static_cast<char>('X' + _bestOBS);

            for (int _obs = 0; _obs <= 2; ++_obs) {
                for (int i: observables_on_iQubit[ith_qubit][_obs]) {
                    if (_bestOBS == _obs) {
                        if (obsMatchCount[i] != numeric_limits<int>::max())
                            obsMatchCount[i]--;
                    } else {
                        obsMatchCount[i] = numeric_limits<int>::max();
                    }
                }
            }
        }

        deRandomScheme.push_back(single_measurement);

        for (int i = 0; i < observableNumber; ++i)
            if (obsMatchCount[i] == 0)
                cur_num_of_measurements[i]++;

        int success = 0;
        for (int i = 0; i < observableNumber; ++i)
            if (cur_num_of_measurements[i] >= floor(observables_weight[i] * measurementTimes_perObservable))
                success++;
        cerr << "[Status " << measurement_repetition + 1 << ": " << success << "]" << endl;

        if (success == observableNumber)
            break;
    }

    if (outputFileName) {
        ofstream outFile(*outputFileName);
        if (outFile.is_open()) {
            for (const auto &measurement: deRandomScheme) {
                for (char c: measurement) {
                    outFile << c << " ";
                }
                outFile << endl;
            }
            outFile.close();
        } else {
            cerr << "Error: Could not open file " << *outputFileName << " for writing." << endl;
        }
    }
    return deRandomScheme;
}

PYBIND11_MODULE(generateMeasureScheme, m) {
    py::class_<MeasureScheme_backend>(m, "MeasureScheme_backend")
            .def(py::init<>())
            .def_readonly("eta", &MeasureScheme_backend::eta)
            .def_readonly("max_k_local", &MeasureScheme_backend::max_k_local)
            .def("randomGenerate", &MeasureScheme_backend::randomGenerate, py::arg("totalMeasurementTimes"),
                 py::arg("outputFileName") = py::none())
            .def("deRandomGenerate", &MeasureScheme_backend::deRandomGenerate,
                 py::arg("measurementTimes_perObservable"),
                 py::arg("observables4CS") = py::none(),
                 py::arg("observablesFileName") = py::none(), py::arg("outputFileName") = py::none());
}