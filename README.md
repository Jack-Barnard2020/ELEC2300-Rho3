# ELEC2300 Rho3 - Neural Network

A C++ implementation of a small multilayer perceptron for the ELEC2300 Rho3 neural-network laboratory. The program implements its own `Matrix` class, trains a neural network as a **4-bit binary decoder**, saves trained weights to disk, and can reload models for prediction and testing.

## Overview

The decoder accepts one of the 16 possible 4-bit binary values and classifies it into one of 16 output classes. Internally, inputs are represented as a four-element column matrix and targets use a 16-element one-hot encoded column matrix.

For example:

```text
Decimal input: 3
Binary input:  0 0 1 1
Target class:  3
```

The network output contains 16 activation values. The output neuron with the largest activation is used as the predicted class.

> **Note on confidence:** the CLI labels the winning sigmoid activation as `Confidence`. It is useful for comparing how strongly the model selects an output, but it is not a calibrated probability.

## Project Structure

```text
ELEC2300-Rho3/
├── Makefile
├── README.md
├── includes/
│   ├── ansi.hpp
│   ├── matrix.hpp
│   └── neural_network.hpp
└── src/
    ├── main.cpp
    ├── matrix.cpp
    └── neural_network.cpp
```

### `includes/matrix.hpp` / `src/matrix.cpp`

Defines and implements the `Matrix` class used throughout the network. It provides:

- matrix addition and subtraction
- matrix multiplication
- scalar multiplication
- matrix transposition
- Hadamard (element-wise) multiplication
- applying an activation function to every element
- mean-squared value calculation for loss reporting
- conversion of a vector into a column matrix
- formatted stream output

### `includes/neural_network.hpp` / `src/neural_network.cpp`

Defines and implements the neural network. It handles:

- configurable network architectures
- sigmoid activation
- forward propagation
- backpropagation and weight updates
- prediction
- model serialisation
- loading previously trained models

### `includes/ansi.hpp`

Contains ANSI escape sequences used by the command-line interface for colours, bold/dim text, clearing the screen, and returning the cursor home.

### `src/main.cpp`

Contains the user-facing command-line interface, training loop, decoder input/target generation, model testing, and command-line argument handling.

## Requirements

The project is written for C++17.

On macOS, Apple's `clang++` can be used through the supplied Makefile. No OpenMP dependency is required by this version of the project.

## Building

From the root of the project:

```bash
make
```

This creates:

```text
./neural_network
```

To remove generated object files and the executable:

```bash
make clean
```

A debug build is also available:

```bash
make debug
```

## Interactive CLI

Start the program without arguments:

```bash
./neural_network
```

The main menu provides:

```text
1. Train a new model
2. Use a saved model
3. Test a saved model
4. Model information
5. Help
6. Quit
```

## Training a Model

Select **Train a new model** from the interactive menu.

The CLI asks for:

1. number of epochs
2. learning rate
3. number of hidden layers
4. neurons per hidden layer
5. reporting interval
6. model filename or path

The decoder input and output sizes are fixed at four and sixteen neurons. Every hidden layer uses the neuron count entered by the user.

For example:

```text
Epochs: 20000
Learning rate: 0.3
Number of hidden layers: 2
Neurons per hidden layer: 16
Report interval: 500
Model filename or path: models/decoder.model
```

This generates the architecture:

```text
4 -> 16 -> 16 -> 16
```

During training, the program reports the mean loss at the chosen interval:

```text
Epoch      1  Loss 0.12345678
Epoch    500  Loss 0.04567890
Epoch   1000  Loss 0.01234567
```

The exact values depend on the model and training run. A useful training run should show loss reducing as training progresses.

### Training from the command line

A model can also be trained without the interactive menu:

```bash
./neural_network train \
    --epochs 20000 \
    --rate 0.3 \
    --architecture 4,16,16,16 \
    --report 500 \
    --model models/decoder.model
```

The architecture supplied with `--architecture` must:

- start with `4`
- contain at least one hidden layer
- end with `16`

Example valid architectures:

```text
4,8,16
4,12,16
4,32,32,16
4,64,64,64,16
```

## Using a Saved Model

From the interactive menu, select:

```text
2. Use a saved model
```

The program first asks for the model filename or path:

```text
Enter model filename or path: models/decoder.model
```

Both relative and absolute paths can be supplied, for example:

```text
decoder.model
models/decoder.model
../models/decoder.model
/Users/username/Documents/ELEC2300-Rho3/models/decoder.model
```

The CLI then asks for a decimal input from `0` to `15` and displays the predicted class and all 16 output activations.

From the command line:

```bash
./neural_network predict 5 --model models/decoder.model
```

## Testing a Saved Model

Select:

```text
3. Test a saved model
```

Enter the filename or path of the model to test. The program evaluates all 16 possible 4-bit inputs and displays:

- input value
- predicted class
- winning output activation (`Confidence`)
- PASS or FAIL

Example layout:

```text
Input    Prediction    Confidence    Result
--------------------------------------------------
    0             0        0.9812    PASS
    1             1        0.9734    PASS
    2             2        0.9641    PASS
    ...
--------------------------------------------------
Correct: 16/16
```

The values above are illustrative only.

Command-line equivalent:

```bash
./neural_network test --model models/decoder.model
```

## Model Information

The model-information command loads a saved model and displays its stored learning rate and architecture.

Interactive menu:

```text
4. Model information
```

Command line:

```bash
./neural_network info --model models/decoder.model
```

Example:

```text
Model file:    models/decoder.model
Learning rate: 0.3
Architecture:  4 -> 16 -> 16 -> 16
```

## Command-Line Reference

```text
./neural_network
./neural_network menu
./neural_network train [options]
./neural_network predict <0-15> [options]
./neural_network test [options]
./neural_network info [options]
./neural_network help
```

Available options:

```text
--epochs N
    Number of training epochs.

--rate R
    Learning rate used during training.

--architecture 4,...,16
    Network architecture when training.

--report N
    Number of epochs between loss reports.

--model FILE
    Model filename or path to save/load.

--no-colour
    Disable ANSI terminal formatting.
```

## How the Neural Network Works

### Forward propagation

Each layer calculates a weighted input using:

```text
z = W * a + b
```

where:

- `W` is the layer's weight matrix
- `a` is the activation from the previous layer
- `b` is the bias matrix

The sigmoid activation function is then applied to each value:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

The resulting activation is passed to the next layer.

### Training and backpropagation

For each training example the program:

1. runs forward propagation
2. compares the predicted output with the one-hot target
3. calculates a loss value
4. works backwards through the network
5. calculates the activation derivative
6. calculates the layer error/gradient
7. calculates changes to the weights and biases
8. applies those changes using the learning rate

The 16 decoder examples are shuffled between training epochs in the current implementation.

### Weight initialisation

Random weights are initialised using a Xavier/Glorot-style uniform range. This helps keep sigmoid inputs in a useful range when the network contains multiple layers.

## Model File Format

Models are stored as plain text.

The file contains:

1. number of network layers
2. size of each layer
3. learning rate
4. weight values for each layer
5. bias values for each layer

The same program can therefore reconstruct the architecture before reading the stored weights and biases.

Model files are intended to be read and written by the program rather than edited manually.

## Suggested Starting Configuration

A useful starting point for the 4-bit decoder is:

```text
Epochs:                  20000
Learning rate:           0.3
Number of hidden layers: 1
Neurons per hidden layer: 12
Report interval:         500
```

Architecture:

```text
4 -> 12 -> 16
```

This is a project recommendation rather than a required laboratory setting. Once the basic model trains correctly, architectures with different hidden-layer counts and widths can be compared.

## Interpreting Results

### Loss

Loss is reported during training to make it easier to see whether the model is learning. A run that remains almost constant for a large number of epochs is worth investigating before simply increasing the model size or epoch count.

### Accuracy

The test command checks all 16 possible 4-bit inputs. `16/16` means the network selected the correct output neuron for every possible decoder input.

### Confidence

`Confidence` is the sigmoid activation of the winning output neuron.

For example:

```text
Prediction: 7
Confidence: 0.96
```

means output neuron 7 had an activation of approximately `0.96` and was the largest output. It should not be interpreted as a calibrated 96% statistical probability.

## Troubleshooting

### `Cannot open model file`

Check that the model exists and supply either the correct relative path or its absolute path.

For example:

```text
models/decoder.model
```

instead of:

```text
decoder.model
```

if the model is stored in the `models` directory.

### Model predicts the same class for every input

Check the training-loss trace. If loss quickly falls and then remains almost unchanged while accuracy stays poor, use a smaller architecture first and confirm the training implementation is behaving correctly.

A useful diagnostic architecture is:

```text
4 -> 12 -> 16
```

### Very deep networks train poorly

This project uses sigmoid activations and a deliberately simple neural-network implementation. Start with one or two hidden layers before experimenting with much deeper networks.

### ANSI codes appear as text

Run the program in a terminal with ANSI escape-code support. ANSI formatting can also be disabled using:

```bash
./neural_network menu --no-colour
```

## Laboratory Context

This project was created for the ELEC2300 Rho3 Neural Networks laboratory. The laboratory covers matrix manipulation in C++, building a simple neural-network model, training a 4-bit binary decoder, monitoring loss, saving/loading trained weights, and experimenting with network architecture.

The laboratory documentation should be treated as the authoritative source for assignment requirements.

## Author

**Jack Barnard**  
ELEC2300 Rho3  
2026
