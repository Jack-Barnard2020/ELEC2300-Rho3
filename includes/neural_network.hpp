#pragma once

#include "matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
