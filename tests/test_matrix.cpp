/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-29
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: test_matrix.cpp
 * File Description: Standalone tests for Matrix calculations and validation
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-29)
 * ======================================================= */

#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Throw on failures so tests still run when the compiler defines NDEBUG.
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectValue(double actual, double expected, const std::string& message) {
    const double scale = std::max(1.0, std::abs(expected));
    require(std::isfinite(actual) && std::abs(actual - expected) <= 1e-12 * scale,
            message + ": expected " + std::to_string(expected)
                    + ", got " + std::to_string(actual));
}

void expectMatrix(const Matrix& actual, const Matrix& expected) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
            "Unexpected matrix dimensions");
    for (std::size_t row = 0; row < expected.rows(); ++row) {
        for (std::size_t col = 0; col < expected.cols(); ++col) {
            expectValue(actual.at(row, col), expected.at(row, col),
                        "Element (" + std::to_string(row) + ", "
                                    + std::to_string(col) + ")");
        }
    }
}

template <class ExpectedException, class Operation>
void expectThrows(Operation operation, const std::string& message) {
    try {
        operation();
    } catch (const ExpectedException&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(message + ": wrong exception: " + error.what());
    }
    throw std::runtime_error(message + ": no exception was thrown");
}

void testConstruction() {
    const Matrix empty;
    require(empty.rows() == 0 && empty.cols() == 0 && empty.data().empty(),
            "Default matrix should be empty");
    const Matrix zero(2, 3);
    expectMatrix(zero, Matrix({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}));
    const Matrix supplied({{1.0, -2.0, 3.5}, {4.0, 5.0, 6.0}});
    require(supplied.rows() == 2 && supplied.cols() == 3,
            "Constructor must preserve a rectangular shape");
    expectValue(supplied.at(0, 2), 3.5, "Constructor must retain values");
    expectThrows<std::invalid_argument>([] {
        Matrix invalid({{1.0, 2.0}, {3.0}});
    }, "Ragged rows must be rejected");
}

void testAccess() {
    Matrix matrix(2, 3);
    matrix.at(1, 2) = 7.5;
    const Matrix& view = matrix;
    expectValue(view.at(1, 2), 7.5, "Const access must see writes");
    expectValue(view.data().at(1).at(2), 7.5, "Data view must expose stored values");
    expectThrows<std::out_of_range>([&] { matrix.at(2, 0) = 1.0; },
                                    "Writable row bounds");
    expectThrows<std::out_of_range>([&] { matrix.at(0, 3) = 1.0; },
                                    "Writable column bounds");
    expectThrows<std::out_of_range>([&] { (void)view.at(2, 0); },
                                    "Const row bounds");
    expectThrows<std::out_of_range>([&] { (void)view.at(0, 3); },
                                    "Const column bounds");
}

void testAdditionAndSubtraction() {
    const Matrix a({{1.0, -2.0, 3.0}, {4.0, 5.0, -6.0}});
    const Matrix b({{6.0, 5.0, 4.0}, {3.0, -2.0, 1.0}});
    expectMatrix(a + b, Matrix({{7.0, 3.0, 7.0}, {7.0, 3.0, -5.0}}));
    expectMatrix(a - b, Matrix({{-5.0, -7.0, -1.0}, {1.0, 7.0, -7.0}}));
    expectMatrix(a, Matrix({{1.0, -2.0, 3.0}, {4.0, 5.0, -6.0}}));
}

void testMatrixProduct() {
    // A non-square example catches swapped row/column and inner-loop indices.
    const Matrix a({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    const Matrix b({{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}});
    expectMatrix(a * b, Matrix({{58.0, 64.0}, {139.0, 154.0}}));
    const Matrix identity({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    expectMatrix(a * identity, a);
    const Matrix column = Matrix::column({1.0, 2.0, 3.0});
    const Matrix dot = column.transpose() * column;
    require(dot.rows() == 1 && dot.cols() == 1, "Dot product dimensions");
    expectValue(dot.at(0, 0), 14.0, "Dot product");
    expectMatrix(a * Matrix(3, 2), Matrix(2, 2));
}

void testScalarProduct() {
    const Matrix a({{1.5, -2.0}, {0.0, 4.0}});
    const Matrix expected({{-3.0, 4.0}, {0.0, -8.0}});
    expectMatrix(a * -2.0, expected);
    expectMatrix(-2.0 * a, expected);
    expectMatrix(a * 0.0, Matrix(2, 2));
    expectMatrix(a, Matrix({{1.5, -2.0}, {0.0, 4.0}}));
}

void testTranspose() {
    const Matrix a({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    expectMatrix(a.transpose(), Matrix({{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}));
    expectMatrix(a.transpose().transpose(), a);
}

void testHadamard() {
    const Matrix a({{1.0, -2.0, 3.0}, {4.0, 5.0, 0.0}});
    const Matrix b({{6.0, 5.0, -4.0}, {3.0, 2.0, 1.0}});
    expectMatrix(a.hadamard(b), Matrix({{6.0, -10.0, -12.0}, {12.0, 10.0, 0.0}}));
}

double square(double value) {
    return value * value;
}

void testApply() {
    const Matrix a({{-2.0, 0.0}, {1.5, 3.0}});
    expectMatrix(a.apply(square), Matrix({{4.0, 0.0}, {2.25, 9.0}}));
    expectMatrix(a, Matrix({{-2.0, 0.0}, {1.5, 3.0}}));
}

void testMeanSquared() {
    const Matrix errors({{-1.0, 2.0}, {-3.0, 4.0}});
    expectValue(errors.meanSquared(), 7.5, "Mean squared error");
    expectValue(Matrix(2, 3).meanSquared(), 0.0, "Zero matrix loss");
    expectThrows<std::logic_error>([] { (void)Matrix().meanSquared(); },
                                  "Loss from an empty matrix");
    expectThrows<std::logic_error>([] { (void)Matrix(2, 0).meanSquared(); },
                                  "Loss from a zero-width matrix");
}

void testColumn() {
    expectMatrix(Matrix::column({1.0, -2.0, 3.5}), Matrix({{1.0}, {-2.0}, {3.5}}));
    const Matrix empty = Matrix::column({});
    require(empty.rows() == 0 && empty.cols() == 1, "Empty column dimensions");
}

void testCopyIndependence() {
    const Matrix original({{1.0, 2.0}, {3.0, 4.0}});
    Matrix copy(original);
    copy.at(0, 0) = 99.0;
    Matrix assigned;
    assigned = original;
    assigned.at(1, 1) = -99.0;
    expectMatrix(original, Matrix({{1.0, 2.0}, {3.0, 4.0}}));
    expectValue(copy.at(0, 0), 99.0, "Copy is writable independently");
    expectValue(assigned.at(1, 1), -99.0, "Assignment is writable independently");
}

void testRandomInitialisation() {
    std::mt19937 first(123u);
    std::mt19937 sameSeed(123u);
    const Matrix a(3, 5, first);
    const Matrix repeat(3, 5, sameSeed);
    expectMatrix(a, repeat);
    // The Xavier range depends on input + output widths, rather than sample count.
    const double limit = std::sqrt(6.0 / 8.0);
    bool hasNonZero = false;
    for (const auto& row : a.data()) {
        for (double value : row) {
            require(std::isfinite(value) && std::abs(value) <= limit,
                    "Random weight must lie within the Xavier range");
            hasNonZero = hasNonZero || value != 0.0;
        }
    }
    require(hasNonZero, "Random initialisation must populate weights");
}

void testShapeValidation() {
    const Matrix a(2, 3);
    const Matrix wrongRows(3, 3);
    const Matrix wrongCols(2, 2);
    // Check each dimension separately; a matching row count alone is insufficient.
    for (const Matrix* other : {&wrongRows, &wrongCols}) {
        expectThrows<std::invalid_argument>([&] { (void)(a + *other); },
                                            "Addition shape mismatch");
        expectThrows<std::invalid_argument>([&] { (void)(a - *other); },
                                            "Subtraction shape mismatch");
        expectThrows<std::invalid_argument>([&] { (void)a.hadamard(*other); },
                                            "Hadamard shape mismatch");
    }
    expectThrows<std::invalid_argument>([&] { (void)(a * wrongCols); },
                                        "Matrix product inner dimension mismatch");
}

void testStreamOutput() {
    const Matrix a({{1.25, -2.5}, {0.0, 3.0}});
    std::ostringstream output;
    output << a;
    const std::string text = output.str();
    require(text.find("1.2500") != std::string::npos
            && text.find("-2.5000") != std::string::npos,
            "Stream output must include formatted values");
    require(std::count(text.begin(), text.end(), '\n') == 2,
            "Stream output should contain one line per row");
}

} // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests = {
        {"Construction", testConstruction},
        {"Access and bounds", testAccess},
        {"Addition and subtraction", testAdditionAndSubtraction},
        {"Matrix product", testMatrixProduct},
        {"Scalar product", testScalarProduct},
        {"Transpose", testTranspose},
        {"Hadamard product", testHadamard},
        {"Element-wise apply", testApply},
        {"Mean squared error", testMeanSquared},
        {"Column construction", testColumn},
        {"Copy independence", testCopyIndependence},
        {"Random initialisation", testRandomInitialisation},
        {"Shape validation", testShapeValidation},
        {"Stream output", testStreamOutput}
    };

    std::size_t failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
        }
    }

    std::cout << '\n' << (tests.size() - failures) << '/' << tests.size()
              << " Matrix tests passed\n";
    return failures == 0 ? 0 : 1;
}
