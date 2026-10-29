/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: matrix.cpp
 * File Description: Shared matrix implementation and interface
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 * ======================================================= */

#include "matrix.hpp"

#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

Matrix::Matrix()
    : rows_(0), cols_(0) {
}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows),
      cols_(cols),
      data_(rows, std::vector<double>(cols, 0.0)) {
}

Matrix::Matrix(
    std::size_t rows,
    std::size_t cols,
    std::mt19937& generator
)
    : Matrix(rows, cols) {

    // Xavier/Glorot uniform initialisation. This keeps sigmoid inputs
    // in a useful range, especially when more than one layer is used.
    const double denominator =
        static_cast<double>((rows + cols) == 0 ? 1 : rows + cols);

    const double limit = std::sqrt(6.0 / denominator);

    std::uniform_real_distribution<double> distribution(
        -limit,
        limit
    );

    for (auto& row : data_) {
        for (double& value : row) {
            value = distribution(generator);
        }
    }
}

Matrix::Matrix(
    const std::vector<std::vector<double>>& values
)
    : rows_(values.size()),
      cols_(values.empty() ? 0 : values.front().size()),
      data_(values) {

    for (const auto& row : data_) {
        if (row.size() != cols_) {
            throw std::invalid_argument(
                "Matrix rows must have equal length"
            );
        }
    }
}

std::size_t Matrix::rows() const noexcept {
    return rows_;
}

std::size_t Matrix::cols() const noexcept {
    return cols_;
}

double& Matrix::at(std::size_t row, std::size_t col) {
    return data_.at(row).at(col);
}

const double& Matrix::at(
    std::size_t row,
    std::size_t col
) const {
    return data_.at(row).at(col);
}

const std::vector<std::vector<double>>&
Matrix::data() const noexcept {
    return data_;
}

void Matrix::requireSameShape(
    const Matrix& rhs,
    const char* operation
) const {
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
        throw std::invalid_argument(
            std::string("Shape mismatch for ") + operation
        );
    }
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    requireSameShape(rhs, "addition");
    Matrix result(rows_, cols_);

    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            result.data_[row][col] =
                data_[row][col] + rhs.data_[row][col];
        }
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    requireSameShape(rhs, "subtraction");
    Matrix result(rows_, cols_);

    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            result.data_[row][col] =
                data_[row][col] - rhs.data_[row][col];
        }
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    if (cols_ != rhs.rows_) {
        throw std::invalid_argument(
            "Shape mismatch for multiplication"
        );
    }

    Matrix result(rows_, rhs.cols_);

    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t inner = 0; inner < cols_; ++inner) {
            for (std::size_t col = 0; col < rhs.cols_; ++col) {
                result.data_[row][col] +=
                    data_[row][inner] * rhs.data_[inner][col];
            }
        }
    }

    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(*this);

    for (auto& row : result.data_) {
        for (double& value : row) {
            value *= scalar;
        }
    }

    return result;
}

Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);

    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            result.data_[col][row] = data_[row][col];
        }
    }

    return result;
}

// Element-wise multiplication is distinct from the matrix product operator*.
Matrix Matrix::hadamard(const Matrix& rhs) const {
    requireSameShape(rhs, "Hadamard product");
    Matrix result(rows_, cols_);

    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            result.data_[row][col] =
                data_[row][col] * rhs.data_[row][col];
        }
    }

    return result;
}

Matrix Matrix::apply(
    double (*function)(double)
) const {
    Matrix result(*this);

    for (auto& row : result.data_) {
        for (double& value : row) {
            value = function(value);
        }
    }

    return result;
}

double Matrix::meanSquared() const {
    if (rows_ == 0 || cols_ == 0) {
        throw std::logic_error(
            "Cannot calculate loss from an empty matrix"
        );
    }

    double sum = 0.0;

    for (const auto& row : data_) {
        for (double value : row) {
            sum += value * value;
        }
    }

    return sum / static_cast<double>(rows_ * cols_);
}

Matrix Matrix::column(
    const std::vector<double>& values
) {
    Matrix result(values.size(), 1);

    for (std::size_t row = 0; row < values.size(); ++row) {
        result.data_[row][0] = values[row];
    }

    return result;
}

std::ostream& operator<<(
    std::ostream& output,
    const Matrix& matrix
) {
    output << std::fixed << std::setprecision(4);

    for (const auto& row : matrix.data()) {
        output << "[ ";

        for (double value : row) {
            output << std::setw(8) << value << ' ';
        }

        output << "]\n";
    }

    return output;
}
