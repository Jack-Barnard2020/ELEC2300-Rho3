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
 *       - Unified terminal theme and in-place screens (2026-10-29)
 * ======================================================= */

#include "ansi.hpp"
#include "terminal.hpp"
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
const std::string SECTION = "Section 3 - 4-bit Decoder";

struct Options {
    std::string modelFile = "neuralNetworks/decoder.model";
    std::size_t epochs = 20000;
    std::size_t reportInterval = 500;
    double learningRate = 0.3;
    std::vector<std::size_t> architecture = {4, 12, 16};
};

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument(message);
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

std::string askForModelFile(const Options& options, const std::string& action) {
    return ui::prompt(SECTION, action, "Model file", "Saved decoder model to load.", options.modelFile);
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
            ui::setEnabled(false);
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
    auto renderProgress = [&](std::size_t epoch, double loss) {
        // One current progress screen replaces the growing epoch transcript.
        if (ui::enabled() || epoch == 0) {
            ui::screen(SECTION, "Training");
            std::cout << "Epochs:          " << options.epochs << '\n'
                      << "Learning rate:   " << options.learningRate << '\n'
                      << "Report interval: " << options.reportInterval << '\n'
                      << "Model file:      " << options.modelFile << '\n'
                      << "Architecture:    ";
            displayArchitecture(options.architecture);
            std::cout << "\n\n";
        }
        if (epoch == 0) {
            std::cout << ui::styled(DIM, "Starting training...") << '\n';
        } else {
            std::cout << ui::styled(CYAN, "Epoch ") << epoch << '/' << options.epochs
                      << "  " << ui::styled(YELLOW, "Loss ")
                      << std::fixed << std::setprecision(8) << loss << '\n';
        }
        std::cout << std::flush;
    };
    renderProgress(0, 0.0);
    NeuralNetwork network(options.architecture, options.learningRate);
    std::array<unsigned, 16> trainingOrder{};
    for (unsigned value = 0; value < trainingOrder.size(); ++value) trainingOrder[value] = value;
    // A fixed seed makes the sample order repeatable between runs.
    std::mt19937 shuffleGenerator(20261023u);
    for (std::size_t epoch = 1; epoch <= options.epochs; ++epoch) {
        std::shuffle(trainingOrder.begin(), trainingOrder.end(), shuffleGenerator);
        double totalLoss = 0.0;
        for (unsigned value : trainingOrder)
            totalLoss += network.update(makeBinaryInput(value), makeTargetOutput(value));
        if (epoch == 1 || epoch % options.reportInterval == 0 || epoch == options.epochs)
            renderProgress(epoch, totalLoss / 16.0);
    }
    const std::filesystem::path path(options.modelFile);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    network.save(options.modelFile);
    std::cout << '\n' << ui::styled(GREEN, "Training complete")
              << "\nSaved model to: " << options.modelFile << '\n';
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

    ui::screen(SECTION, "Prediction");

    std::cout
        << "\nModel:      " << options.modelFile
        << "\nInput:      " << inputValue
        << "\nPrediction: "
        << ui::styled(prediction == inputValue ? GREEN : RED,
            std::to_string(prediction)
        )
        << "\n\n";
    ui::heading("Output scores");

    for (std::size_t index = 0; index < output.rows(); ++index) {
        std::cout
            << (index == prediction ? ui::styled(CYAN, "> ") : "  ")
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

    ui::screen(SECTION, "Test Results");

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
            << ui::styled(correct ? GREEN : RED, correct ? "PASS" : "FAIL")
            << '\n';
    }

    std::cout
        << "--------------------------------------------------\n"
        << ui::styled(BOLD, "Correct: ")
        << correctPredictions << "/16\n";
}

void displayModelInformation(const Options& options) {
    NeuralNetwork network = NeuralNetwork::load(options.modelFile);
    if (network.architecture().front() != 4 || network.architecture().back() != 16)
        fail("This is not a decoder model (expected 4 inputs and 16 outputs)");
    ui::screen(SECTION, "Model Information");

    std::cout
        << "\nModel file:    " << options.modelFile
        << "\nLearning rate: " << network.learningRate()
        << "\nArchitecture:  ";
    displayArchitecture(network.architecture());
    std::cout << '\n';
}

void displayHelp() {
    ui::screen(SECTION, "Help");
    ui::heading("Usage");
    std::cout
        << "  ./builds/neural_network decoder\n"
        << "  ./builds/neural_network decoder train [options]\n"
        << "  ./builds/neural_network decoder predict <0-15> [options]\n"
        << "  ./builds/neural_network decoder test [options]\n"
        << "  ./builds/neural_network decoder info [options]\n\n";
    ui::heading("Options");
    std::cout
        << "  --epochs N\n"
        << "  --rate R\n"
        << "  --architecture 4,12,16\n"
        << "  --report N\n"
        << "  --model FILE (default: neuralNetworks/decoder.model)\n"
        << "  --no-colour (disable ANSI colours and screen clearing)\n";
}

void readTrainingSettings(Options& options) {
    const std::string action = "Train a New Model";
    options.epochs = cli::size(ui::prompt(SECTION, action, "Epochs",
        "One epoch trains on all 16 inputs.", std::to_string(options.epochs)));
    options.learningRate = cli::rate(ui::prompt(SECTION, action, "Learning rate",
        "Positive size of each weight update.", std::to_string(options.learningRate)));
    const auto hiddenLayerCount = cli::size(ui::prompt(SECTION, action, "Number of hidden layers",
        "Use at least one hidden layer.", std::to_string(options.architecture.size() - 2)));
    const auto neuronsPerHiddenLayer = cli::size(ui::prompt(SECTION, action, "Neurons per hidden layer",
        "Each hidden layer will use this width.", std::to_string(options.architecture.at(1))));
    if (hiddenLayerCount == 0 || neuronsPerHiddenLayer == 0)
        fail("The network needs at least one non-empty hidden layer");
    options.architecture.assign(hiddenLayerCount, neuronsPerHiddenLayer);
    options.architecture.insert(options.architecture.begin(), 4);
    options.architecture.push_back(16);
    options.reportInterval = cli::size(ui::prompt(SECTION, action, "Report interval",
        "Refresh training progress after this many epochs.", std::to_string(options.reportInterval)));
    options.modelFile = ui::prompt(SECTION, action, "Model output",
        "Where to save the trained weights and biases.", options.modelFile);
    if (options.epochs == 0 || options.reportInterval == 0 || options.modelFile.empty())
        fail("Training settings are invalid");
}

void runMenu(Options options) {
    while (std::cin) {
        ui::screen(SECTION, "Decoder Menu");
        ui::menuItem(1, "Train a new model");
        ui::menuItem(2, "Use a saved model");
        ui::menuItem(3, "Test a saved model");
        ui::menuItem(4, "Model information");
        ui::menuItem(5, "Help");
        ui::menuItem(6, "Quit");
        ui::choicePrompt();
        std::string line;
        if (!std::getline(std::cin, line)) return;
        try {
            const auto choice = cli::size(line);
            if (choice == 1) {
                readTrainingSettings(options);
                trainModel(options);
            } else if (choice == 2) {
                options.modelFile = askForModelFile(options, "Use a Saved Model");
                const auto value = cli::size(ui::prompt(SECTION, "Use a Saved Model", "Input (0-15)",
                    "Decimal value to encode as four binary inputs.", "0"));
                if (value > 15) fail("Input must be between 0 and 15");
                predictValue(options, static_cast<unsigned>(value));
            } else if (choice == 3) {
                options.modelFile = askForModelFile(options, "Test a Saved Model");
                testModel(options);
            } else if (choice == 4) {
                options.modelFile = askForModelFile(options, "Model Information");
                displayModelInformation(options);
            } else if (choice == 5) {
                displayHelp();
            } else if (choice == 6) {
                ui::screen(SECTION, "Menu Closed");
                return;
            } else {
                fail("Choose a number from 1 to 6.");
            }
        } catch (const std::exception& error) {
            ui::error(SECTION, error.what());
        }
        ui::waitForEnter();
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
            displayHelp();
        } else {
            fail("Unknown command: " + command);
        }

        return 0;
    } catch (const std::exception& error) {
        ui::error(SECTION, error.what());
        return 1;
    }
}
