/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: decoder.cpp
 * File Description: Section 3 decoder commands and interactive menu
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 *       - Updated executable and model folder paths (2026-10-29)
 * ======================================================= */

#include "ansi.hpp"
#include "cli.hpp"
#include "commands.hpp"
#include <filesystem>
#include "neural_network.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ansi;

struct Options {
    std::string modelFile = "neuralNetworks/decoder.model";
    std::size_t epochs = 20000;
    std::size_t reportInterval = 500;
    double learningRate = 0.3;
    std::vector<std::size_t> architecture = {4, 12, 16};
    bool useColour = true;
};

std::string coloured(
    const Options& options,
    const char* code,
    const std::string& text
) {
    return options.useColour
        ? std::string(code) + text + RESET
        : text;
}

void clearScreen(const Options& options) {
    if (options.useColour) {
        std::cout << CLEAR_SCREEN << CURSOR_HOME;
    }
}

void displayBanner(const Options& options) {
    std::cout
        << coloured(options, CYAN, "========================================\n")
        << coloured(options, BOLD, "       ELEC2300 Rho3 Neural Network\n")
        << coloured(options, CYAN, "========================================\n");
}

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument(message);
}

void waitForEnter(const Options& options) {
    std::cout << coloured(options, DIM, "\nPress Enter to continue...");
    std::string line;
    std::getline(std::cin, line);
}

std::string readLine() {
    std::string line;
    if (!std::getline(std::cin, line)) fail("Input ended");
    return line;
}

void displayArchitecture(
    const std::vector<std::size_t>& architecture
) {
    for (std::size_t layer = 0; layer < architecture.size(); ++layer) {
        if (layer > 0) {
            std::cout << " -> ";
        }
        std::cout << architecture[layer];
    }
}

std::vector<std::size_t> parseArchitecture(
    const std::string& text
) {
    std::vector<std::size_t> architecture;
    std::stringstream stream(text);
    std::string layerText;

    while (std::getline(stream, layerText, ',')) {
        if (layerText.empty()) {
            fail("Architecture contains an empty layer");
        }

        architecture.push_back(cli::size(layerText));
        if (architecture.back() == 0) fail("Layer sizes must be positive");
    }
    if (!text.empty() && text.back() == ',') fail("Architecture contains an empty layer");

    if (
        architecture.size() < 3 ||
        architecture.front() != 4 ||
        architecture.back() != 16
    ) {
        fail(
            "Architecture must start with 4, contain a hidden layer, "
            "and end with 16"
        );
    }

    return architecture;
}

std::string askForModelFile(const Options& options) {
    std::string fileName;
    std::cout
        << '\n'
        << coloured(options, CYAN, "Model File\n")
        << "----------------------------------------\n"
        << "Enter model filename or path: ";

    fileName = readLine();
    if (fileName.empty()) {
        fail("A model filename or path is required");
    }
    return fileName;
}

std::string nextArgumentValue(
    int& index,
    int argc,
    char* argv[],
    const std::string& option
) {
    ++index;
    if (index >= argc) {
        fail("Missing value after " + option);
    }
    return argv[index];
}

void parseOptions(
    Options& options,
    int startIndex,
    int argc,
    char* argv[]
) {
    for (int index = startIndex; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--model") {
            options.modelFile = nextArgumentValue(index, argc, argv, argument);
        } else if (argument == "--epochs") {
            const std::string value =
                nextArgumentValue(index, argc, argv, argument);
            if (value.empty() || value.front() == '-') {
                fail("Epoch count must be a non-negative integer");
            }
            options.epochs = cli::size(value);
        } else if (argument == "--rate") {
            options.learningRate = cli::rate(
                nextArgumentValue(index, argc, argv, argument)
            );
        } else if (argument == "--architecture") {
            options.architecture = parseArchitecture(
                nextArgumentValue(index, argc, argv, argument)
            );
        } else if (argument == "--report") {
            options.reportInterval = cli::size(
                nextArgumentValue(index, argc, argv, argument)
            );
        } else if (argument == "--no-colour") {
            options.useColour = false;
        } else {
            fail("Unknown option: " + argument);
        }
    }

    if (options.epochs == 0) {
        fail("Epoch count must be positive");
    }
    if (options.learningRate <= 0.0) {
        fail("Learning rate must be positive");
    }
    if (options.reportInterval == 0) {
        fail("Report interval must be positive");
    }
}

// Inputs are MSB-first column vectors: 3 becomes [0, 0, 1, 1].
Matrix makeBinaryInput(unsigned value) {
    if (value > 15) {
        fail("Input must be between 0 and 15");
    }

    std::vector<double> binary(4, 0.0);
    for (std::size_t bit = 0; bit < binary.size(); ++bit) {
        binary[3 - bit] = static_cast<double>((value >> bit) & 1U);
    }
    return Matrix::column(binary);
}

// Output index equals the decoded integer (one-hot encoding).
Matrix makeTargetOutput(unsigned value) {
    if (value > 15) {
        fail("Target must be between 0 and 15");
    }

    std::vector<double> target(16, 0.0);
    target.at(value) = 1.0;
    return Matrix::column(target);
}

std::size_t findPrediction(const Matrix& output) {
    if (output.rows() == 0 || output.cols() != 1) {
        fail("Invalid neural network output");
    }

    std::size_t best = 0;
    for (std::size_t index = 1; index < output.rows(); ++index) {
        if (output.at(index, 0) > output.at(best, 0)) {
            best = index;
        }
    }
    return best;
}

void trainModel(const Options& options) {
    clearScreen(options);
    displayBanner(options);

    NeuralNetwork network(
        options.architecture,
        options.learningRate
    );

    std::cout
        << '\n'
        << coloured(options, YELLOW, "Training Configuration\n")
        << "----------------------------------------\n"
        << "Epochs:          " << options.epochs << '\n'
        << "Learning rate:   " << options.learningRate << '\n'
        << "Report interval: " << options.reportInterval << '\n'
        << "Model file:      " << options.modelFile << '\n'
        << "Architecture:    ";

    displayArchitecture(options.architecture);
    std::cout << "\n----------------------------------------\n\n";

    std::array<unsigned, 16> trainingOrder{};
    for (unsigned value = 0; value < trainingOrder.size(); ++value) {
        trainingOrder[value] = value;
    }

    // A fixed seed makes the sample order repeatable between runs.
    std::mt19937 shuffleGenerator(20261023u);

    for (std::size_t epoch = 1; epoch <= options.epochs; ++epoch) {
        std::shuffle(
            trainingOrder.begin(),
            trainingOrder.end(),
            shuffleGenerator
        );

        double totalLoss = 0.0;
        for (unsigned value : trainingOrder) {
            totalLoss += network.update(
                makeBinaryInput(value),
                makeTargetOutput(value)
            );
        }

        const bool shouldReport =
            epoch == 1 ||
            epoch % options.reportInterval == 0 ||
            epoch == options.epochs;

        if (shouldReport) {
            std::cout
                << coloured(options, CYAN, "Epoch ")
                << std::setw(6) << epoch
                << "  "
                << coloured(options, YELLOW, "Loss ")
                << std::fixed << std::setprecision(8)
                << totalLoss / 16.0
                << '\n';
        }
    }

    const std::filesystem::path path(options.modelFile);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    network.save(options.modelFile);
    std::cout
        << '\n'
        << coloured(options, GREEN, "Training complete\n")
        << "Saved model to: " << options.modelFile << '\n';
}

void predictValue(
    const Options& options,
    unsigned inputValue
) {
    NeuralNetwork network = NeuralNetwork::load(options.modelFile);
    if (network.architecture().front() != 4 || network.architecture().back() != 16)
        fail("This is not a decoder model (expected 4 inputs and 16 outputs)");
    const Matrix output = network.predict(makeBinaryInput(inputValue));
    const std::size_t prediction = findPrediction(output);

    clearScreen(options);
    displayBanner(options);

    std::cout
        << "\nModel:      " << options.modelFile
        << "\nInput:      " << inputValue
        << "\nPrediction: "
        << coloured(
            options,
            prediction == inputValue ? GREEN : RED,
            std::to_string(prediction)
        )
        << "\n\nOutput scores\n";

    for (std::size_t index = 0; index < output.rows(); ++index) {
        std::cout
            << (index == prediction ? coloured(options, CYAN, "> ") : "  ")
            << std::setw(2) << index
            << "  " << std::fixed << std::setprecision(6)
            << output.at(index, 0)
            << '\n';
    }
}

void testModel(const Options& options) {
    NeuralNetwork network = NeuralNetwork::load(options.modelFile);
    if (network.architecture().front() != 4 || network.architecture().back() != 16)
        fail("This is not a decoder model (expected 4 inputs and 16 outputs)");
    std::size_t correctPredictions = 0;

    clearScreen(options);
    displayBanner(options);

    std::cout
        << "\nModel: " << options.modelFile << "\n\n"
        << "Input    Prediction    Confidence    Result\n"
        << "--------------------------------------------------\n";

    for (unsigned value = 0; value < 16; ++value) {
        const Matrix output = network.predict(makeBinaryInput(value));
        const std::size_t predicted = findPrediction(output);
        const double confidence = output.at(predicted, 0);
        const bool correct = predicted == value;

        if (correct) {
            ++correctPredictions;
        }

        std::cout
            << std::setw(5) << value << "    "
            << std::setw(10) << predicted << "    "
            << std::fixed << std::setprecision(4)
            << std::setw(10) << confidence << "    "
            << coloured(options, correct ? GREEN : RED, correct ? "PASS" : "FAIL")
            << '\n';
    }

    std::cout
        << "--------------------------------------------------\n"
        << coloured(options, BOLD, "Correct: ")
        << correctPredictions << "/16\n";
}

void displayModelInformation(const Options& options) {
    NeuralNetwork network = NeuralNetwork::load(options.modelFile);
    if (network.architecture().front() != 4 || network.architecture().back() != 16)
        fail("This is not a decoder model (expected 4 inputs and 16 outputs)");
    clearScreen(options);
    displayBanner(options);

    std::cout
        << "\nModel file:    " << options.modelFile
        << "\nLearning rate: " << network.learningRate()
        << "\nArchitecture:  ";
    displayArchitecture(network.architecture());
    std::cout << '\n';
}

void displayHelp(const Options& options) {
    clearScreen(options);
    displayBanner(options);
    std::cout
        << "\nUsage:\n"
        << "  ./builds/neural_network decoder\n"
        << "  ./builds/neural_network decoder train [options]\n"
        << "  ./builds/neural_network decoder predict <0-15> [options]\n"
        << "  ./builds/neural_network decoder test [options]\n"
        << "  ./builds/neural_network decoder info [options]\n\n"
        << "Options:\n"
        << "  --epochs N\n"
        << "  --rate R\n"
        << "  --architecture 4,12,16\n"
        << "  --report N\n"
        << "  --model FILE (default: neuralNetworks/decoder.model)\n"
        << "  --no-colour\n";
}

void readTrainingSettings(Options& options) {
    clearScreen(options);
    displayBanner(options);

    std::cout << "\nEpochs: ";
    options.epochs = cli::size(readLine());

    std::cout << "Learning rate: ";
    options.learningRate = cli::rate(readLine());

    std::size_t hiddenLayerCount = 0;
    std::cout << "Number of hidden layers: ";
    hiddenLayerCount = cli::size(readLine());

    std::size_t neuronsPerHiddenLayer = 0;
    std::cout << "Neurons per hidden layer: ";
    neuronsPerHiddenLayer = cli::size(readLine());

    if (hiddenLayerCount == 0 || neuronsPerHiddenLayer == 0) {
        fail("The network needs at least one non-empty hidden layer");
    }

    options.architecture.clear();
    options.architecture.push_back(4);
    for (std::size_t layer = 0; layer < hiddenLayerCount; ++layer) {
        options.architecture.push_back(neuronsPerHiddenLayer);
    }
    options.architecture.push_back(16);

    std::cout << "Report interval: ";
    options.reportInterval = cli::size(readLine());

    std::cout << "Model filename or path: ";
    options.modelFile = readLine();

    if (
        options.epochs == 0 ||
        options.learningRate <= 0.0 ||
        options.reportInterval == 0 ||
        options.modelFile.empty()
    ) {
        fail("Training settings are invalid");
    }
}

void runMenu(Options options) {
    while (std::cin) {
        clearScreen(options);
        displayBanner(options);

        std::cout
            << "\n1. Train a new model\n"
            << "2. Use a saved model\n"
            << "3. Test a saved model\n"
            << "4. Model information\n"
            << "5. Help\n"
            << "6. Quit\n"
            << "Choice: ";

        std::string line;
        if (!std::getline(std::cin, line)) return;
        try {
            const auto choice = cli::size(line);
            if (choice == 1) {
                readTrainingSettings(options);
                trainModel(options);
            } else if (choice == 2) {
                options.modelFile = askForModelFile(options);
                unsigned value = 0;
                std::cout << "Input (0-15): ";
                value = cli::size(readLine());
                predictValue(options, value);
            } else if (choice == 3) {
                options.modelFile = askForModelFile(options);
                testModel(options);
            } else if (choice == 4) {
                options.modelFile = askForModelFile(options);
                displayModelInformation(options);
            } else if (choice == 5) {
                displayHelp(options);
            } else if (choice == 6) {
                return;
            } else {
                std::cout << coloured(options, RED, "Invalid choice\n");
            }
        } catch (const std::exception& error) {
            std::cerr
                << coloured(
                    options,
                    RED,
                    std::string("Error: ") + error.what()
                )
                << '\n';
        }

        waitForEnter(options);
    }
}

} // namespace

int runDecoder(int argc, char* argv[]) {
    try {
        if (argc == 1) {
            runMenu(Options{});
            return 0;
        }

        const std::string command = argv[1];
        Options options;

        if (command == "predict") {
            if (argc < 3) {
                fail("Predict requires an input from 0 to 15");
            }
            const auto parsedValue = cli::size(argv[2]);
            if (parsedValue > 15) {
                fail("Input must be between 0 and 15");
            }
            parseOptions(options, 3, argc, argv);
            predictValue(options, static_cast<unsigned>(parsedValue));
        } else if (command == "train") {
            parseOptions(options, 2, argc, argv);
            trainModel(options);
        } else if (command == "test") {
            parseOptions(options, 2, argc, argv);
            testModel(options);
        } else if (command == "info") {
            parseOptions(options, 2, argc, argv);
            displayModelInformation(options);
        } else if (command == "menu") {
            parseOptions(options, 2, argc, argv);
            runMenu(options);
        } else if (
            command == "help" ||
            command == "--help" ||
            command == "-h"
        ) {
            parseOptions(options, 2, argc, argv);
            displayHelp(options);
        } else {
            fail("Unknown command: " + command);
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
