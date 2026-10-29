/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: matrix.hpp
 * File Description: Shared matrix implementation and interface
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 * ======================================================= */

#pragma once

#include <cstddef>
#include <iosfwd>
#include <random>
#include <vector>

// Dense row-major matrix. Network activations are N-by-1 columns.
class Matrix {
public:
    Matrix();
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, std::mt19937& generator);
    explicit Matrix(const std::vector<std::vector<double>>& values);

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;

    double& at(std::size_t row, std::size_t col);
    const double& at(std::size_t row, std::size_t col) const;
    const std::vector<std::vector<double>>& data() const noexcept;

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator*(const Matrix& rhs) const;
    Matrix operator*(double scalar) const;

    Matrix transpose() const;
    Matrix hadamard(const Matrix& rhs) const;
    Matrix apply(double (*function)(double)) const;
    double meanSquared() const;

    static Matrix column(const std::vector<double>& values);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::vector<double>> data_;

    void requireSameShape(const Matrix& rhs, const char* operation) const;
};

Matrix operator*(double scalar, const Matrix& matrix);
std::ostream& operator<<(std::ostream& output, const Matrix& matrix);
