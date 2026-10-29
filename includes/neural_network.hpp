/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: neural_network.hpp
 * File Description: Shared neural network implementation and interface
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 * ======================================================= */

#pragma once

#include "matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Fully connected sigmoid layers; weights[l] has next-by-previous dimensions.
class NeuralNetwork {
public:
    NeuralNetwork(
        const std::vector<std::size_t>& architecture,
        double learningRate,
        std::uint32_t seed = 5489u
    );

    Matrix predict(const Matrix& input) const;
    double update(const Matrix& input, const Matrix& target);

    void save(const std::string& fileName) const;
    static NeuralNetwork load(const std::string& fileName);

    const std::vector<std::size_t>& architecture() const noexcept;
    double learningRate() const noexcept;

private:
    std::vector<std::size_t> architecture_;
    double learningRate_;
    std::vector<Matrix> weights_;
    std::vector<Matrix> biases_;

    static double sigmoid(double value);
    static double derivative(double output);
    std::vector<Matrix> forward(const Matrix& input) const;
};
