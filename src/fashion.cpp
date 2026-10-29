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
 * ======================================================= */

#include "ansi.hpp"
#include "cli.hpp"
#include "commands.hpp"
#include "neural_network.hpp"
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
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
    std::cout << ansi::BOLD << "Actual label: " << sample.label << " - " << CLASS_NAMES[sample.label]
              << ansi::RESET << "\n\n";
    for (std::size_t row = 0; row < 28; ++row) {
        for (std::size_t col = 0; col < 28; ++col)
            std::cout << shades[(static_cast<unsigned>(sample.raw[row * 28 + col]) * 9) / 255];
        std::cout << '\n';
    }
}

void train(const Options& options) {
    std::cout << ansi::CYAN << ansi::BOLD << "\nTRAINING\n" << ansi::RESET
              << "Dataset:      " << locateCsv(options.trainCsv) << '\n'
              << "Architecture: " << joinArchitecture(options.architecture) << '\n'
              << "Epochs:       " << options.epochs << '\n'
              << "Learning rate:" << ' ' << options.rate << '\n'
              << "Sample limit: " << (options.limit ? std::to_string(options.limit) : "all rows") << "\n\n";
    NeuralNetwork network(options.architecture, options.rate, 20261023u);
    for (std::size_t epoch = 1; epoch <= options.epochs; ++epoch) {
        double loss = 0.0;
        std::size_t count = readCsv(options.trainCsv, options.limit, [&](const Sample& sample, std::size_t index) {
            loss += network.update(sample.input, sample.target);
            if ((index + 1) % options.report == 0)
                std::cout << ansi::DIM << "Epoch " << epoch << ": " << (index + 1)
                          << " samples, mean loss " << loss / (index + 1) << ansi::RESET << '\n';
        });
        std::cout << ansi::GREEN << "Epoch " << epoch << " complete" << ansi::RESET
                  << ": " << count << " samples, mean loss " << loss / count << '\n';
    }
    const std::filesystem::path modelPath(options.model);
    if (modelPath.has_parent_path()) std::filesystem::create_directories(modelPath.parent_path());
    network.save(options.model);
    std::cout << ansi::GREEN << ansi::BOLD << "Saved model: " << options.model << ansi::RESET << '\n';
}

void test(const Options& options) {
    std::cout << ansi::CYAN << ansi::BOLD << "\nTESTING\n" << ansi::RESET
              << "Dataset: " << locateCsv(options.testCsv) << '\n'
              << "Model:   " << options.model << "\n\n";
    auto network = NeuralNetwork::load(options.model);
    if (network.architecture().front() != PIXELS || network.architecture().back() != CLASSES)
        fail("This is not a Fashion-MNIST model (expected 784 inputs and 10 outputs)");
    std::array<std::array<std::size_t, CLASSES>, CLASSES> confusion{};
    std::size_t correct = 0;
    bool shown = false;
    const std::size_t count = readCsv(options.testCsv, options.limit, [&](const Sample& sample, std::size_t) {
        const Matrix output = network.predict(sample.input);
        const std::size_t prediction = argmax(output);
        ++confusion[sample.label][prediction];
        if (prediction == sample.label) ++correct;
        if (options.preview && !shown) {
            showImage(sample);
            std::cout << "Prediction: " << prediction << " - " << CLASS_NAMES[prediction]
                      << " (output " << output.at(prediction, 0) << ")\n\n";
            shown = true;
        }
    });
    const double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(count);
    std::cout << ansi::GREEN << ansi::BOLD << "Accuracy: " << std::fixed << std::setprecision(2)
              << accuracy << "%" << ansi::RESET << " (" << correct << '/' << count << ")\n\n";
    std::cout << "Confusion matrix: rows = actual class, columns = predicted class\n    ";
    for (std::size_t col = 0; col < CLASSES; ++col) std::cout << std::setw(5) << col;
    std::cout << '\n';
    for (std::size_t row = 0; row < CLASSES; ++row) {
        std::cout << std::setw(3) << row << ' ';
        for (auto value : confusion[row]) std::cout << std::setw(5) << value;
        std::cout << "  " << CLASS_NAMES[row] << '\n';
    }
}

void predictOne(const Options& options, std::size_t index) {
    auto network = NeuralNetwork::load(options.model);
    if (network.architecture().front() != PIXELS || network.architecture().back() != CLASSES)
        fail("This is not a Fashion-MNIST model (expected 784 inputs and 10 outputs)");
    bool found = false;
    readCsv(options.testCsv, index + 1, [&](const Sample& sample, std::size_t current) {
        if (current != index) return;
        found = true;
        const Matrix output = network.predict(sample.input);
        const std::size_t prediction = argmax(output);
        showImage(sample);
        std::cout << "\nPredicted: " << ansi::BOLD << prediction << " - " << CLASS_NAMES[prediction]
                  << ansi::RESET << "\n\nOutputs:\n";
        for (std::size_t i = 0; i < CLASSES; ++i)
            std::cout << i << "  " << std::setw(12) << CLASS_NAMES[i] << "  " << output.at(i, 0) << '\n';
    });
    if (!found) fail("The requested image index is beyond the test CSV.");
}

void modelInfo(const Options& options) {
    const auto network = NeuralNetwork::load(options.model);
    std::cout << ansi::CYAN << ansi::BOLD << "\nMODEL INFORMATION\n" << ansi::RESET
              << "File:          " << options.model << '\n'
              << "Architecture:  " << joinArchitecture(network.architecture()) << '\n'
              << "Learning rate: " << network.learningRate() << '\n';
}

void printHelp() {
    std::cout << ansi::CYAN << ansi::BOLD << "\nELEC2300 Rho3 - Fashion-MNIST optional work\n" << ansi::RESET
              << "Run without arguments for the interactive menu.\n\n"
              << ansi::BOLD << "Commands\n" << ansi::RESET
              << "  train          Train a new network using the training CSV and save its weights.\n"
              << "  test           Load a saved model, test it against the test CSV, and print accuracy.\n"
              << "  predict INDEX  Display one test image and predict its clothing class. INDEX starts at 0.\n"
              << "  init           Create an untrained model file using the chosen architecture.\n"
              << "  info           Show the architecture and learning rate stored in a model.\n"
              << "  help           Show this explanation.\n\n"
              << ansi::BOLD << "Options\n" << ansi::RESET
              << "  --train-csv FILE       Training CSV path. The program also checks ./data and ./datasets.\n"
              << "  --test-csv FILE        Test CSV path.\n"
              << "  --model FILE           Model weights file to save or load (default: neuralNetworks/fashion.model).\n"
              << "  --epochs N             Number of complete passes through the selected training data.\n"
              << "  --rate R               Learning rate used when updating weights (default: 0.05).\n"
              << "  --architecture LIST    Layer sizes, e.g. 784,64,10. 784 inputs and 10 outputs are fixed.\n"
              << "  --limit N              Only use the first N rows. 0 means use every row. Useful for quick tests.\n"
              << "  --report N             Print training loss after every N samples.\n"
              << "  --preview              During testing, render the first 28x28 image in the terminal.\n\n"
              << ansi::BOLD << "Examples\n" << ansi::RESET
              << "  ./builds/neural_network fashion train --train-csv fashion-mnist_train.csv --epochs 3\n"
              << "  ./builds/neural_network fashion test --test-csv fashion-mnist_test.csv --preview\n"
              << "  ./builds/neural_network fashion train --architecture 784,128,10 --model neuralNetworks/fashion-128.model\n";
}

std::string promptText(const std::string& title, const std::string& explanation, const std::string& current) {
    std::cout << ansi::BOLD << title << ansi::RESET << "\n" << ansi::DIM << explanation << ansi::RESET
              << "\nCurrent: " << current << "\nEnter new value, or press Enter to keep it: ";
    std::string line;
    if (!std::getline(std::cin, line)) fail("Input ended");
    return line.empty() ? current : line;
}

std::size_t promptSize(const std::string& title, const std::string& explanation, std::size_t current, bool zeroAllowed) {
    const std::string value = promptText(title, explanation, std::to_string(current));
    const auto parsed = cli::size(value);
    if (!zeroAllowed && parsed == 0) fail(title + " must be greater than zero.");
    return parsed;
}

double promptRate(double current) {
    const std::string value = promptText("Learning rate", "Controls the size of each weight update. Keep 0.05 for the baseline experiment.", std::to_string(current));
    const double parsed = cli::rate(value);
    if (parsed <= 0.0) fail("Learning rate must be positive.");
    return parsed;
}

void waitForEnter() {
    std::cout << ansi::DIM << "\nPress Enter to return to the menu..." << ansi::RESET;
    std::string line;
    std::getline(std::cin, line);
}

void clearScreen() { std::cout << ansi::CLEAR_SCREEN << ansi::CURSOR_HOME; }

void interactiveMenu() {
    Options options;
    while (std::cin) {
        clearScreen();
        std::cout << ansi::CYAN << ansi::BOLD
                  << "==================================================\n"
                  << "       ELEC2300 Rho3 - Fashion-MNIST CLI\n"
                  << "==================================================\n" << ansi::RESET
                  << "This program completes the optional clothing-classification work.\n"
                  << "Choose an action below. You can press Enter at prompts to keep defaults.\n\n"
                  << ansi::GREEN << "1) Train model" << ansi::RESET << "      Read training images and update the network weights.\n"
                  << ansi::GREEN << "2) Test model" << ansi::RESET << "       Measure classification accuracy on unseen test images.\n"
                  << ansi::GREEN << "3) Predict image" << ansi::RESET << "    Inspect one test image and the network's prediction.\n"
                  << ansi::YELLOW << "4) Initialise model" << ansi::RESET << " Create a model with random weights without training it.\n"
                  << ansi::CYAN << "5) Model information" << ansi::RESET << " Show the saved model's architecture and learning rate.\n"
                  << "6) Help                Explain every command and option.\n"
                  << "7) Quit\n\n"
                  << ansi::BOLD << "Choice: " << ansi::RESET;
        std::string line;
        if (!std::getline(std::cin, line)) return;
        try {
            const auto choice = cli::size(line);
            if (choice == 1) {
                clearScreen();
                std::cout << ansi::CYAN << ansi::BOLD << "TRAIN MODEL\n\n" << ansi::RESET;
                options.trainCsv = promptText("Training CSV", "Path to the Fashion-MNIST training CSV. You may enter just the filename; ./data and ./datasets are also searched.", options.trainCsv);
                options.model = promptText("Model output", "Where trained weights and biases will be saved.", options.model);
                options.epochs = promptSize("Epochs", "One epoch is one complete pass through the selected training rows.", options.epochs, false);
                options.rate = promptRate(options.rate);
                const std::string architecture = promptText("Architecture", "Comma-separated neurons in each layer. Fashion-MNIST requires 784 inputs and 10 outputs; change the hidden layers for section 4.2.", joinArchitecture(options.architecture));
                options.architecture = parseArchitecture(architecture);
                options.limit = promptSize("Training row limit", "Use fewer rows for a quick check. Enter 0 to use the entire CSV.", options.limit, true);
                options.report = promptSize("Loss report interval", "Print average training loss after this many samples.", options.report, false);
                train(options);
            } else if (choice == 2) {
                clearScreen();
                std::cout << ansi::CYAN << ansi::BOLD << "TEST MODEL\n\n" << ansi::RESET;
                options.testCsv = promptText("Test CSV", "Path to the Fashion-MNIST test CSV. You may enter just the filename; ./data and ./datasets are also searched.", options.testCsv);
                options.model = promptText("Model file", "Saved model to evaluate.", options.model);
                options.limit = promptSize("Test row limit", "Use fewer rows for a quick check. Enter 0 to test every row.", options.limit, true);
                const std::string preview = promptText("Preview image", "Enter y to render the first test image using terminal characters, otherwise n.", options.preview ? "y" : "n");
                options.preview = !preview.empty() && (preview[0] == 'y' || preview[0] == 'Y');
                test(options);
            } else if (choice == 3) {
                clearScreen();
                std::cout << ansi::CYAN << ansi::BOLD << "PREDICT ONE IMAGE\n\n" << ansi::RESET;
                options.testCsv = promptText("Test CSV", "CSV containing the image you want to inspect.", options.testCsv);
                options.model = promptText("Model file", "Saved model used to make the prediction.", options.model);
                const std::size_t index = promptSize("Image index", "Zero-based row index: 0 selects the first image after the CSV header.", 0, true);
                predictOne(options, index);
            } else if (choice == 4) {
                clearScreen();
                options.model = promptText("Model output", "File that will receive the randomly initialised network.", options.model);
                const std::string architecture = promptText("Architecture", "Example: 784,64,10. The middle values are hidden layers.", joinArchitecture(options.architecture));
                options.architecture = parseArchitecture(architecture);
                options.rate = promptRate(options.rate);
                const std::filesystem::path modelPath(options.model);
                if (modelPath.has_parent_path()) std::filesystem::create_directories(modelPath.parent_path());
                NeuralNetwork(options.architecture, options.rate, 20261023u).save(options.model);
                std::cout << ansi::GREEN << "Initialised " << options.model << ansi::RESET << '\n';
            } else if (choice == 5) {
                clearScreen();
                options.model = promptText("Model file", "Saved model whose configuration you want to inspect.", options.model);
                modelInfo(options);
            } else if (choice == 6) {
                clearScreen();
                printHelp();
            } else if (choice == 7) {
                return;
            } else {
                std::cout << ansi::YELLOW << "Please choose a number from 1 to 7." << ansi::RESET << '\n';
            }
        } catch (const std::exception& error) {
            std::cerr << ansi::RED << ansi::BOLD << "\nError: " << error.what() << ansi::RESET << '\n';
        }
        waitForEnter();
    }
}
}

int runFashion(int argc, char** argv) {
    try {
        if (argc < 2) {
            interactiveMenu();
            return 0;
        }
        const std::string command = argv[1];
        if (command == "predict") {
            if (argc < 3) fail("predict requires an image index. Example: predict 0");
            const std::size_t index = cli::size(argv[2]);
            predictOne(parseOptions(3, argc, argv), index);
        } else if (command == "train") train(parseOptions(2, argc, argv));
        else if (command == "test") test(parseOptions(2, argc, argv));
        else if (command == "init") {
            const Options options = parseOptions(2, argc, argv);
            const std::filesystem::path modelPath(options.model);
            if (modelPath.has_parent_path()) std::filesystem::create_directories(modelPath.parent_path());
            NeuralNetwork(options.architecture, options.rate, 20261023u).save(options.model);
            std::cout << ansi::GREEN << "Initialised " << options.model << ansi::RESET << '\n';
        } else if (command == "info") modelInfo(parseOptions(2, argc, argv));
        else if (command == "menu") interactiveMenu();
        else if (command == "help" || command == "--help" || command == "-h") printHelp();
        else fail("Unknown command: " + command + ". Use 'help' to see available commands.");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << ansi::RED << ansi::BOLD << "Error: " << error.what() << ansi::RESET << '\n';
        return 1;
    }
}
