/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: fashion.cpp
 * File Description: Section 4 Fashion-MNIST commands, CSV reader and menu
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
#include "neural_network.hpp"
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const std::string SECTION = "Section 4 - Fashion-MNIST";
constexpr std::size_t PIXELS = 784;
constexpr std::size_t CLASSES = 10;
const std::array<const char*, CLASSES> CLASS_NAMES = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

struct Options {
    std::string trainCsv = "fashion-mnist_train.csv";
    std::string testCsv = "fashion-mnist_test.csv";
    std::string model = "neuralNetworks/fashion.model";
    std::vector<std::size_t> architecture = {784, 64, 10};
    std::size_t epochs = 3;
    std::size_t limit = 0;
    std::size_t report = 1000;
    double rate = 0.05;
    bool preview = false;
};

struct Sample {
    Matrix input;
    Matrix target;
    std::size_t label;
    std::array<unsigned char, PIXELS> raw;
};

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

std::string joinArchitecture(const std::vector<std::size_t>& a) {
    std::ostringstream out;
    for (std::size_t i = 0; i < a.size(); ++i) out << (i ? "," : "") << a[i];
    return out.str();
}

std::vector<std::size_t> parseArchitecture(const std::string& text) {
    std::vector<std::size_t> architecture;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const auto value = cli::size(token);
        if (value == 0) fail("Invalid architecture. Example: 784,64,10");
        architecture.push_back(value);
    }
    if (!text.empty() && text.back() == ',') fail("Architecture contains an empty layer");
    if (architecture.size() < 3 || architecture.front() != PIXELS || architecture.back() != CLASSES)
        fail("Fashion-MNIST architecture must start with 784, contain at least one hidden layer, and end with 10.");
    return architecture;
}

std::string optionValue(int& index, int argc, char** argv) {
    if (++index >= argc) fail(std::string("Missing value after ") + argv[index - 1]);
    return argv[index];
}

Options parseOptions(int start, int argc, char** argv) {
    Options options;
    for (int i = start; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--train-csv") options.trainCsv = optionValue(i, argc, argv);
        else if (argument == "--test-csv") options.testCsv = optionValue(i, argc, argv);
        else if (argument == "--model") options.model = optionValue(i, argc, argv);
        else if (argument == "--epochs") options.epochs = cli::size(optionValue(i, argc, argv));
        else if (argument == "--limit") options.limit = cli::size(optionValue(i, argc, argv));
        else if (argument == "--report") options.report = cli::size(optionValue(i, argc, argv));
        else if (argument == "--rate") options.rate = cli::rate(optionValue(i, argc, argv));
        else if (argument == "--architecture") options.architecture = parseArchitecture(optionValue(i, argc, argv));
        else if (argument == "--preview") options.preview = true;
        else fail("Unknown option: " + argument + ". Use 'help' to see all options.");
    }
    if (options.epochs == 0 || options.report == 0 || options.rate <= 0.0)
        fail("Epochs, report interval and learning rate must be positive.");
    return options;
}

std::string locateCsv(const std::string& requested) {
    namespace fs = std::filesystem;
    const fs::path input(requested);
    std::vector<fs::path> candidates;
    candidates.push_back(input);
    if (!input.is_absolute()) {
        candidates.push_back(fs::path("data") / input);
        candidates.push_back(fs::path("datasets") / input);
        candidates.push_back(fs::path("Optional Work") / input);
        if (input.has_parent_path()) candidates.push_back(input.filename());
    }
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec))
            return candidate.string();
    }
    std::ostringstream error;
    error << "Cannot open CSV '" << requested << "'.\n"
          << "Working directory: " << fs::current_path().string() << "\n"
          << "Looked for:";
    for (const auto& candidate : candidates) error << "\n  - " << candidate.string();
    error << "\nEnter the CSV path in the interactive menu, or use --train-csv/--test-csv.";
    throw std::runtime_error(error.str());
}

// Normalise pixels to [0, 1] and construct a one-hot target column.
Sample parseRow(const std::string& line, std::size_t lineNumber) {
    if (!line.empty() && line.back() == ',') fail("Trailing empty CSV field");
    std::stringstream stream(line);
    std::string field;
    std::vector<int> values;
    while (std::getline(stream, field, ',')) {
        if (field.empty()) fail("Empty CSV field at line " + std::to_string(lineNumber));
        std::size_t used = 0;
        const int value = std::stoi(field, &used);
        while (used < field.size() && std::isspace(static_cast<unsigned char>(field[used]))) ++used;
        if (used != field.size()) fail("Invalid integer at line " + std::to_string(lineNumber));
        values.push_back(value);
    }
    if (values.size() != PIXELS + 1)
        fail("Expected 785 comma-separated values at line " + std::to_string(lineNumber));
    if (values[0] < 0 || values[0] >= static_cast<int>(CLASSES))
        fail("Label outside 0-9 at line " + std::to_string(lineNumber));

    std::vector<double> input(PIXELS);
    std::vector<double> target(CLASSES, 0.0);
    std::array<unsigned char, PIXELS> raw{};
    for (std::size_t i = 0; i < PIXELS; ++i) {
        const int pixel = values[i + 1];
        if (pixel < 0 || pixel > 255) fail("Pixel outside 0-255 at line " + std::to_string(lineNumber));
        raw[i] = static_cast<unsigned char>(pixel);
        input[i] = static_cast<double>(pixel) / 255.0;
    }
    target[values[0]] = 1.0;
    return {Matrix::column(input), Matrix::column(target), static_cast<std::size_t>(values[0]), raw};
}

// Stream rows on each epoch instead of retaining the whole dataset in RAM.
// The lab CSV format requires a header, which is consumed before sample zero.
template <class Consumer>
std::size_t readCsv(const std::string& requested, std::size_t limit, Consumer consume) {
    const std::string fileName = locateCsv(requested);
    std::ifstream input(fileName);
    if (!input) throw std::runtime_error("Cannot open CSV: " + fileName);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("CSV is empty: " + fileName);
    std::size_t count = 0, lineNumber = 1;
    while (std::getline(input, line) && (limit == 0 || count < limit)) {
        ++lineNumber;
        if (line.empty()) continue;
        consume(parseRow(line, lineNumber), count);
        ++count;
    }
    if (input.bad()) throw std::runtime_error("CSV read failed: " + fileName);
    if (count == 0) throw std::runtime_error("CSV contains no data rows: " + fileName);
    return count;
}

std::size_t argmax(const Matrix& output) {
    if (output.cols() != 1 || output.rows() == 0) fail("Invalid neural-network output shape.");
    std::size_t best = 0;
    for (std::size_t i = 1; i < output.rows(); ++i)
        if (output.at(i, 0) > output.at(best, 0)) best = i;
    return best;
}

void showImage(const Sample& sample) {
    const char* shades = " .:-=+*#%@";
    std::cout << ui::styled(ansi::CYAN, "Actual label: ") << sample.label
              << " - " << CLASS_NAMES[sample.label] << "\n\n";
    for (std::size_t row = 0; row < 28; ++row) {
        for (std::size_t col = 0; col < 28; ++col)
            std::cout << shades[(static_cast<unsigned>(sample.raw[row * 28 + col]) * 9) / 255];
        std::cout << '\n';
    }
}

void trainingConfiguration(const Options& options) {
    ui::screen(SECTION, "Training");
    std::cout << "Dataset:         " << options.trainCsv << '\n'
              << "Architecture:    " << joinArchitecture(options.architecture) << '\n'
              << "Epochs:          " << options.epochs << '\n'
              << "Learning rate:   " << options.rate << '\n'
              << "Report interval: " << options.report << " samples\n"
              << "Model file:      " << options.model << '\n'
              << "Sample limit:    " << (options.limit ? std::to_string(options.limit) : "all rows")
              << "\n\n";
}

void train(const Options& options) {
    trainingConfiguration(options);
    // Resolve paths and construct the network after rendering, so errors replace this page.
    locateCsv(options.trainCsv);
    NeuralNetwork network(options.architecture, options.rate, 20261023u);
    for (std::size_t epoch = 1; epoch <= options.epochs; ++epoch) {
        double loss = 0.0;
        auto renderProgress = [&](std::size_t count, bool complete) {
            if (ui::enabled()) trainingConfiguration(options);
            std::cout << ui::styled(ansi::CYAN, "Epoch ") << epoch << '/' << options.epochs
                      << "  Samples: " << count << "  " << ui::styled(ansi::YELLOW, "Loss ")
                      << std::fixed << std::setprecision(8) << loss / count;
            if (complete) std::cout << "  " << ui::styled(ansi::GREEN, "Epoch complete");
            std::cout << '\n' << std::flush;
        };
        const std::size_t count = readCsv(options.trainCsv, options.limit,
            [&](const Sample& sample, std::size_t index) {
                loss += network.update(sample.input, sample.target);
                if ((index + 1) % options.report == 0) renderProgress(index + 1, false);
            });
        renderProgress(count, true);
    }
    const std::filesystem::path modelPath(options.model);
    if (modelPath.has_parent_path()) std::filesystem::create_directories(modelPath.parent_path());
    network.save(options.model);
    std::cout << '\n' << ui::styled(ansi::GREEN, "Training complete")
              << "\nSaved model to: " << options.model << '\n';
}

void test(const Options& options) {
    ui::screen(SECTION, "Testing");
    std::cout << "Dataset: " << options.testCsv << "\nModel:   " << options.model << '\n';
    auto network = NeuralNetwork::load(options.model);
    if (network.architecture().front() != PIXELS || network.architecture().back() != CLASSES)
        fail("This is not a Fashion-MNIST model (expected 784 inputs and 10 outputs)");
    std::array<std::array<std::size_t, CLASSES>, CLASSES> confusion{};
    std::size_t correct = 0;
    // Keep only the first preview so progress refreshes cannot erase it permanently.
    std::optional<Sample> preview;
    const std::size_t count = readCsv(options.testCsv, options.limit,
        [&](const Sample& sample, std::size_t index) {
            const Matrix output = network.predict(sample.input);
            const std::size_t prediction = argmax(output);
            ++confusion[sample.label][prediction];
            if (prediction == sample.label) ++correct;
            if (options.preview && !preview) preview = sample;
            if ((index + 1) % options.report == 0) {
                if (ui::enabled()) ui::screen(SECTION, "Testing");
                std::cout << "Model: " << options.model << '\n'
                          << ui::styled(ansi::CYAN, "Samples tested: ") << index + 1 << '\n'
                          << "Correct: " << correct << '\n' << std::flush;
            }
        });
    ui::screen(SECTION, "Test Results");
    const double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(count);
    std::cout << "Model: " << options.model << '\n'
              << ui::styled(ansi::GREEN, "Accuracy: ") << std::fixed << std::setprecision(2)
              << accuracy << "% (" << correct << '/' << count << ")\n\n";
    ui::heading("Confusion matrix");
    std::cout << "Rows = actual class, columns = predicted class\n    ";
    for (std::size_t col = 0; col < CLASSES; ++col) std::cout << std::setw(5) << col;
    std::cout << '\n';
    for (std::size_t row = 0; row < CLASSES; ++row) {
        std::cout << std::setw(3) << row << ' ';
        for (auto value : confusion[row]) std::cout << std::setw(5) << value;
        std::cout << "  " << CLASS_NAMES[row] << '\n';
    }
    if (preview) {
        std::cout << '\n';
        ui::heading("First test image");
        showImage(*preview);
        const Matrix output = network.predict(preview->input);
        const std::size_t prediction = argmax(output);
        const std::string result = std::to_string(prediction) + " - " + CLASS_NAMES[prediction];
        std::cout << "Prediction: "
                  << ui::styled(prediction == preview->label ? ansi::GREEN : ansi::RED, result)
                  << " (output " << output.at(prediction, 0) << ")\n";
    }
}

void predictOne(const Options& options, std::size_t index) {
    ui::screen(SECTION, "Prediction");
    auto network = NeuralNetwork::load(options.model);
    if (network.architecture().front() != PIXELS || network.architecture().back() != CLASSES)
        fail("This is not a Fashion-MNIST model (expected 784 inputs and 10 outputs)");
    bool found = false;
    readCsv(options.testCsv, index + 1, [&](const Sample& sample, std::size_t current) {
        if (current != index) return;
        found = true;
        const Matrix output = network.predict(sample.input);
        const std::size_t prediction = argmax(output);
        std::cout << "Model: " << options.model << "\nImage index: " << index << "\n\n";
        showImage(sample);
        const std::string result = std::to_string(prediction) + " - " + CLASS_NAMES[prediction];
        std::cout << "\nPrediction: "
                  << ui::styled(prediction == sample.label ? ansi::GREEN : ansi::RED, result)
                  << "\n\n";
        ui::heading("Output scores");
        for (std::size_t i = 0; i < CLASSES; ++i)
            std::cout << (i == prediction ? ui::styled(ansi::CYAN, "> ") : "  ")
                      << i << "  " << std::setw(12) << CLASS_NAMES[i] << "  "
                      << std::fixed << std::setprecision(6) << output.at(i, 0) << '\n';
    });
    if (!found) fail("The requested image index is beyond the test CSV.");
}

void initialise(const Options& options) {
    ui::screen(SECTION, "Initialise Model");
    const std::filesystem::path modelPath(options.model);
    if (modelPath.has_parent_path()) std::filesystem::create_directories(modelPath.parent_path());
    NeuralNetwork(options.architecture, options.rate, 20261023u).save(options.model);
    std::cout << "Architecture:  " << joinArchitecture(options.architecture) << '\n'
              << "Learning rate: " << options.rate << "\n\n"
              << ui::styled(ansi::GREEN, "Model initialised") << "\nSaved model to: " << options.model << '\n';
}

void modelInfo(const Options& options) {
    ui::screen(SECTION, "Model Information");
    const auto network = NeuralNetwork::load(options.model);
    std::cout << "Model file:    " << options.model << '\n'
              << "Architecture:  " << joinArchitecture(network.architecture()) << '\n'
              << "Learning rate: " << network.learningRate() << '\n';
}

void printHelp() {
    ui::screen(SECTION, "Help");
    std::cout << "Run './builds/neural_network fashion' for the interactive menu.\n\n";
    ui::heading("Commands");
    std::cout << "  train          Train a new network and save its weights.\n"
              << "  test           Evaluate a saved model and print accuracy.\n"
              << "  predict INDEX  Display one test image and predict its class (index starts at 0).\n"
              << "  init           Save an untrained model.\n"
              << "  info           Show a saved model's architecture and learning rate.\n"
              << "  menu           Open the interactive menu.\n"
              << "  help           Show this explanation.\n\n";
    ui::heading("Options");
    std::cout << "  --train-csv FILE       Training CSV (default: fashion-mnist_train.csv).\n"
              << "  --test-csv FILE        Test CSV (default: fashion-mnist_test.csv).\n"
              << "  --model FILE           Model (default: neuralNetworks/fashion.model).\n"
              << "  --epochs N             Training passes (default: 3).\n"
              << "  --rate R               Learning rate (default: 0.05).\n"
              << "  --architecture LIST    Layer sizes (default: 784,64,10).\n"
              << "  --limit N              First N samples for train/test; 0 = all (default: 0).\n"
              << "  --report N             Refresh progress every N samples (default: 1000).\n"
              << "  --preview              Display the first image in the test results.\n"
              << "  --no-colour            Disable ANSI colours and screen clearing.\n\n";
    ui::heading("Examples");
    std::cout << "  ./builds/neural_network fashion train --epochs 3\n"
              << "  ./builds/neural_network fashion test --preview\n"
              << "  ./builds/neural_network fashion train --architecture 784,128,10\n";
}

std::string promptText(const std::string& action, const std::string& title,
                       const std::string& explanation, const std::string& current) {
    return ui::prompt(SECTION, action, title, explanation, current);
}

std::size_t promptSize(const std::string& action, const std::string& title,
                      const std::string& explanation, std::size_t current, bool zeroAllowed) {
    const auto value = cli::size(promptText(action, title, explanation, std::to_string(current)));
    if (!zeroAllowed && value == 0) fail(title + " must be greater than zero.");
    return value;
}

double promptRate(const std::string& action, double current) {
    return cli::rate(promptText(action, "Learning rate",
        "Positive size of each weight update.", std::to_string(current)));
}

void interactiveMenu(Options options = {}) {
    while (std::cin) {
        ui::screen(SECTION, "Fashion-MNIST Menu");
        ui::menuItem(1, "Train a new model");
        ui::menuItem(2, "Test a saved model");
        ui::menuItem(3, "Predict an image");
        ui::menuItem(4, "Initialise a model");
        ui::menuItem(5, "Model information");
        ui::menuItem(6, "Help");
        ui::menuItem(7, "Quit");
        ui::choicePrompt();
        std::string line;
        if (!std::getline(std::cin, line)) return;
        try {
            const auto choice = cli::size(line);
            if (choice == 1) {
                const std::string action = "Train a New Model";
                options.trainCsv = promptText(action, "Training CSV",
                    "Training CSV path. Also searches data/, datasets/ and Optional Work/.", options.trainCsv);
                options.model = promptText(action, "Model output", "Where to save trained weights and biases.", options.model);
                options.epochs = promptSize(action, "Epochs", "Passes through the selected training rows.", options.epochs, false);
                options.rate = promptRate(action, options.rate);
                options.architecture = parseArchitecture(promptText(action, "Architecture",
                    "Comma-separated layer sizes with 784 inputs and 10 outputs.", joinArchitecture(options.architecture)));
                options.limit = promptSize(action, "Training row limit", "0 uses the entire CSV.", options.limit, true);
                options.report = promptSize(action, "Report interval", "Refresh progress after this many samples.", options.report, false);
                train(options);
            } else if (choice == 2) {
                const std::string action = "Test a Saved Model";
                options.testCsv = promptText(action, "Test CSV", "CSV of images to evaluate.", options.testCsv);
                options.model = promptText(action, "Model file", "Saved model to evaluate.", options.model);
                options.limit = promptSize(action, "Test row limit", "0 tests every row.", options.limit, true);
                const std::string preview = promptText(action, "Preview image", "Enter y to display the first image, or n.", options.preview ? "y" : "n");
                options.preview = preview == "y" || preview == "Y";
                test(options);
            } else if (choice == 3) {
                const std::string action = "Predict an Image";
                options.testCsv = promptText(action, "Test CSV", "CSV containing the image to inspect.", options.testCsv);
                options.model = promptText(action, "Model file", "Saved model used for the prediction.", options.model);
                const auto index = promptSize(action, "Image index", "Zero-based data row: 0 is the first image.", 0, true);
                predictOne(options, index);
            } else if (choice == 4) {
                const std::string action = "Initialise a Model";
                options.model = promptText(action, "Model output", "Where to save the untrained network.", options.model);
                options.architecture = parseArchitecture(promptText(action, "Architecture",
                    "Comma-separated layer sizes with 784 inputs and 10 outputs.", joinArchitecture(options.architecture)));
                options.rate = promptRate(action, options.rate);
                initialise(options);
            } else if (choice == 5) {
                options.model = promptText("Model Information", "Model file", "Saved model to inspect.", options.model);
                modelInfo(options);
            } else if (choice == 6) {
                printHelp();
            } else if (choice == 7) {
                ui::screen(SECTION, "Menu Closed");
                return;
            } else {
                fail("Choose a number from 1 to 7.");
            }
        } catch (const std::exception& error) {
            ui::error(SECTION, error.what());
        }
        ui::waitForEnter();
    }
}
} // namespace

int runFashion(int argc, char** argv) {
    try {
        if (argc < 2) {
            interactiveMenu();
            return 0;
        }
        const std::string command = argv[1];
        if (command == "predict") {
            if (argc < 3) fail("predict requires an image index. Example: predict 0");
            predictOne(parseOptions(3, argc, argv), cli::size(argv[2]));
        } else if (command == "train") train(parseOptions(2, argc, argv));
        else if (command == "test") test(parseOptions(2, argc, argv));
        else if (command == "init") initialise(parseOptions(2, argc, argv));
        else if (command == "info") modelInfo(parseOptions(2, argc, argv));
        else if (command == "menu") interactiveMenu(parseOptions(2, argc, argv));
        else if (command == "help" || command == "--help" || command == "-h") printHelp();
        else fail("Unknown command: " + command + ". Use 'help' to see available commands.");
        return 0;
    } catch (const std::exception& error) {
        ui::error(SECTION, error.what());
        return 1;
    }
}
