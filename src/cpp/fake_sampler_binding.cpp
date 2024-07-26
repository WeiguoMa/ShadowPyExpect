#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <map>
#include <string>
#include <bitset>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

class FakeSampler_backend {
private:
    vector<string> proj_basis;
    std::map<string, MatrixXcd> _bases = {
            {"X", (MatrixXcd(2, 2) << 1, 1, -1, 1).finished() / sqrt(2.0)},
            {"Y", (MatrixXcd(2, 2) << 1, -1i, 1i, 1).finished() / sqrt(2.0)},
            {"Z", (MatrixXcd(2, 2) << 1, 0, 0, 1).finished()}
    };

    std::random_device rd;
    std::mt19937 gen;

    template<typename MatrixType>
    static MatrixType kroneckerProduct(const MatrixType& A, const MatrixType& B) {
        const int rowsA = A.rows();
        const int colsA = A.cols();
        const int rowsB = B.rows();
        const int colsB = B.cols();

        MatrixType result(rowsA * rowsB, colsA * colsB);

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsA; ++j) {
                result.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
            }
        }
        return result;
    }
public:
    FakeSampler_backend(vector<string>& basis): proj_basis(basis), gen(rd()) {}

    std::vector<std::string> fakeSampling_dm(py::array_t<std::complex<double>> dm_array,
                                             std::vector<string>& measurement_orientation);
};


std::vector<std::string> FakeSampler_backend::fakeSampling_dm(py::array_t<std::complex<double>> dm_array,
                                         std::vector<string>& measurement_orientation) {
    auto dm_buf = dm_array.request();

    const int rows = dm_buf.shape[0];
    const int cols = dm_buf.shape[1];
    int _systemSize = static_cast<int>(log2(rows));

    Eigen::Map<MatrixXcd> dm(static_cast<std::complex<double>*>(dm_buf.ptr), rows, cols);

    MatrixXcd _U_operations = _bases.at(measurement_orientation[0]);
    for (size_t i = 1; i < measurement_orientation.size(); ++i) {
        _U_operations = kroneckerProduct(_U_operations, _bases.at(measurement_orientation[i]));
    }

    MatrixXcd _operated_DM = _U_operations * dm * _U_operations.adjoint();

    Eigen::VectorXcd _prob = _operated_DM.diagonal();

    std::vector<double> _prob_real;
    for (int i = 0; i < _prob.size(); ++i) {
        _prob_real.push_back(_prob(i).real());
    }

    std::discrete_distribution<> dist(_prob_real.begin(), _prob_real.end());

    std::string _state = proj_basis[dist(gen)];

    std::vector<std::string> _state_eigenValue;
    _state_eigenValue.reserve(_state.size());
    for (char _value : _state) {
        _state_eigenValue.push_back(_value == '0' ? "1" : "-1");
    }

    return _state_eigenValue;
}

PYBIND11_MODULE(fakeSampler_backend, m) {
    py::class_<FakeSampler_backend>(m, "FakeSampler_backend")
            .def(py::init<vector<string>&>(), py::arg("proj_basis"))
            .def("fakeSampling_dm", &FakeSampler_backend::fakeSampling_dm,
                 py::arg("dm_array"), py::arg("measurement_orientation"));
}
