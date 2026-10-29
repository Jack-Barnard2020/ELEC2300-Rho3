/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: neural_network.cpp
 * File Description: Shared neural network implementation and interface
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 * ======================================================= */

#include "neural_network.hpp"

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(
    const std::vector<std::size_t>& architecture,
    double learningRate,
    std::uint32_t seed
)
    : architecture_(architecture),
      learningRate_(learningRate) {

    if (architecture.size() < 2) {
        throw std::invalid_argument(
            "Neural network must have at least two layers"
        );
    }

    if (!std::isfinite(learningRate) || learningRate <= 0.0) {
        throw std::invalid_argument(
            "Learning rate must be positive and finite"
        );
    }

    for (std::size_t layerSize : architecture) {
        if (layerSize == 0) {
            throw std::invalid_argument(
                "Layer sizes must be positive"
            );
        }
    }

    std::mt19937 generator(seed);

    for (
        std::size_t layer = 1;
        layer < architecture.size();
        ++layer
    ) {
        weights_.emplace_back(
            architecture[layer],
            architecture[layer - 1],
            generator
        );

        biases_.emplace_back(
            architecture[layer],
            1
        );
    }
}

// Separate branches avoid overflowing exp() for large negative inputs.
double NeuralNetwork::sigmoid(double value) {
    if (value >= 0.0) {
        const double exponent = std::exp(-value);
        return 1.0 / (1.0 + exponent);
    }

    const double exponent = std::exp(value);
    return exponent / (1.0 + exponent);
}

double NeuralNetwork::derivative(double output) {
    return output * (1.0 - output);
}

std::vector<Matrix> NeuralNetwork::forward(
    const Matrix& input
) const {
    if (
        input.rows() != architecture_.front() ||
        input.cols() != 1
    ) {
        throw std::invalid_argument(
            "Input shape does not match network input layer"
        );
    }

    std::vector<Matrix> activations;
    activations.reserve(architecture_.size());
    activations.push_back(input);

    for (
        std::size_t layer = 0;
        layer < weights_.size();
        ++layer
    ) {
        Matrix weightedInput =
            weights_[layer] * activations.back()
            + biases_[layer];

        activations.push_back(
            weightedInput.apply(sigmoid)
        );
    }

    return activations;
}

Matrix NeuralNetwork::predict(
    const Matrix& input
) const {
    return forward(input).back();
}

double NeuralNetwork::update(
    const Matrix& input,
    const Matrix& target
) {
    if (
        target.rows() != architecture_.back() ||
        target.cols() != 1
    ) {
        throw std::invalid_argument(
            "Target shape does not match network output layer"
        );
    }

    std::vector<Matrix> activations = forward(input);

    Matrix error = target - activations.back();
    const double loss = error.meanSquared();

    for (
        std::size_t layer = weights_.size();
        layer-- > 0;
    ) {
        // Calculate the unscaled layer delta. Keeping the learning rate
        // out of the value propagated backwards prevents it being applied
        // repeatedly across deep networks.
        Matrix layerDelta =
            error.hadamard(
                activations[layer + 1].apply(derivative)
            );

        // Propagate through the weights as they existed before this update.
        Matrix previousError =
            weights_[layer].transpose() * layerDelta;

        Matrix scaledDelta =
            layerDelta * learningRate_;

        Matrix deltaWeights =
            scaledDelta * activations[layer].transpose();

        weights_[layer] =
            weights_[layer] + deltaWeights;

        biases_[layer] =
            biases_[layer] + scaledDelta;

        if (layer > 0) {
            error = previousError;
        }
    }

    return loss;
}

void NeuralNetwork::save(
    const std::string& fileName
) const {
    std::ofstream output(fileName);

    if (!output) {
        throw std::runtime_error(
            "Cannot write model file: " + fileName
        );
    }

    // Text layout: layer count, sizes, rate, then weights and biases per layer.
    // Preserve enough digits to round-trip doubles and existing section models.
    output.precision(17);
    output << architecture_.size() << '\n';

    for (std::size_t layerSize : architecture_) {
        output << layerSize << ' ';
    }

    output << '\n' << learningRate_ << '\n';

    for (
        std::size_t layer = 0;
        layer < weights_.size();
        ++layer
    ) {
        for (const auto& row : weights_[layer].data()) {
            for (double value : row) {
                output << value << ' ';
            }
        }

        output << '\n';

        for (const auto& row : biases_[layer].data()) {
            for (double value : row) {
                output << value << ' ';
            }
        }

        output << '\n';
    }

    if (!output) {
        throw std::runtime_error(
            "Failed while writing model file: " + fileName
        );
    }
}

NeuralNetwork NeuralNetwork::load(
    const std::string& fileName
) {
    std::ifstream input(fileName);

    if (!input) {
        throw std::runtime_error(
            "Cannot open model file: " + fileName
        );
    }

    std::size_t layerCount = 0;

    if (!(input >> layerCount) || layerCount < 2) {
        throw std::runtime_error(
            "Invalid model file: " + fileName
        );
    }

    std::vector<std::size_t> architecture(layerCount);

    for (std::size_t& layerSize : architecture) {
        input >> layerSize;
    }

    double learningRate = 0.0;
    input >> learningRate;

    if (!input || learningRate <= 0.0) {
        throw std::runtime_error(
            "Invalid model metadata: " + fileName
        );
    }

    NeuralNetwork network(architecture, learningRate);

    for (
        std::size_t layer = 0;
        layer < network.weights_.size();
        ++layer
    ) {
        for (
            std::size_t row = 0;
            row < network.weights_[layer].rows();
            ++row
        ) {
            for (
                std::size_t col = 0;
                col < network.weights_[layer].cols();
                ++col
            ) {
                input >> network.weights_[layer].at(row, col);
            }
        }

        for (
            std::size_t row = 0;
            row < network.biases_[layer].rows();
            ++row
        ) {
            input >> network.biases_[layer].at(row, 0);
        }
    }

    if (!input) {
        throw std::runtime_error(
            "Model file is incomplete: " + fileName
        );
    }

    return network;
}

const std::vector<std::size_t>&
NeuralNetwork::architecture() const noexcept {
    return architecture_;
}

double NeuralNetwork::learningRate() const noexcept {
    return learningRate_;
}
