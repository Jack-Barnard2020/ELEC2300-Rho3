# ELEC2300 RHo3 — Sections 3 and 4

Author: **Jack Barnard**

One C++17 command-line application for the section 3 four-bit decoder and
section 4 Fashion-MNIST classifier. Both modes share the same `Matrix` and
`NeuralNetwork` classes. The existing banner comment style and author credit
are retained throughout the source.

## Build and launch

Requires GNU Make and a C++17 compiler with `<filesystem>` support (GCC or
Clang). No third-party C++ libraries are required. On Windows, use WSL or a
suitable MinGW environment.

Extract this project, open a terminal in its directory, then run:

```sh
make
./builds/neural_network
```

The menu selects section 3 or section 4. Quitting a section returns to the
main menu. You can also run commands directly:

```sh
./builds/neural_network help
./builds/neural_network decoder help
./builds/neural_network fashion help
```

`section3` and `section4` are aliases for `decoder` and `fashion`.
`./builds/neural_network decoder` and `./builds/neural_network fashion` open the respective
section menus. `make run` builds and opens the main menu; `make clean` removes
the generated executables and intermediate objects, retaining datasets and saved weights.
Use `make CXX=clang++` to select Clang. Header dependencies are tracked
automatically. Run `make clean` before rebuilding with different compiler flags.

## Folder structure and Makefiles

| Folder | Contents |
| --- | --- |
| `src/` | Application `.cpp` files |
| `includes/` | Shared `.hpp` headers |
| `builds/` | Generated executables: `neural_network` and `test_matrix` |
| `builds/obj/` | Generated object files and header dependency files |
| `neuralNetworks/` | Default saved model weights: `decoder.model` and `fashion.model` |
| `tests/` | C++ test sources and the test Makefile |

The project-root `Makefile` defines the compilation and linking rules.
`tests/Makefile` forwards to those rules, so it also works from the tests folder.
Run application commands from the project root so relative data and weight paths
resolve consistently. `--model FILE` can override the default weight path; put
custom models under `neuralNetworks/` to keep the same layout. The archive
includes empty `builds/` and `neuralNetworks/` directories; executables and
weights are generated when you build or train.

| Command (from the project root) | Result |
| --- | --- |
| `make` or `make app` | Build `builds/neural_network` |
| `make run` | Build and run the interactive CLI |
| `make matrix-test` | Build `builds/test_matrix` |
| `make test` | Build and run the Matrix tests |
| `make -C tests` | Build the Matrix test executable |
| `make -C tests test` | Build and run the Matrix tests via the tests Makefile |
| `make clean` | Remove generated executables, objects and dependency files |

`make -C tests clean` forwards to the same root clean target, so it also removes
the CLI executable. Both executables have independent `main()` functions; the
test links only `tests/test_matrix.cpp` and `src/matrix.cpp`.

## Matrix class tests

```sh
make test
# Alternatively, from the tests directory:
cd tests
make test
```

The standalone `tests/test_matrix.cpp` uses no third-party test framework.
It checks construction, zero initialisation, element access, const access,
bounds checks, addition, subtraction, rectangular matrix multiplication,
scalar multiplication in both orders, transpose, Hadamard multiplication,
function application, mean squared error, column vectors, copying, random
initialisation and stream output. Invalid dimensions and empty loss inputs
are checked for the expected exceptions. Each test prints PASS or FAIL;
the program exits with status 1 if any test fails. Tests remain active when
compiled with `-DNDEBUG`.

## Section 3: four-bit decoder

```sh
./builds/neural_network decoder train --model neuralNetworks/decoder.model --no-colour
./builds/neural_network decoder test --model neuralNetworks/decoder.model --no-colour
./builds/neural_network decoder predict 7 --model neuralNetworks/decoder.model --no-colour
./builds/neural_network decoder info --model neuralNetworks/decoder.model --no-colour
```

Training uses all 16 input values, shuffled each epoch. An input is encoded
most-significant bit first: 3 becomes `[0, 0, 1, 1]`. The target is a 16-element
one-hot vector with index 3 set to 1. Internally these are column matrices.
The predicted value is the index of the largest output activation.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model FILE` | `neuralNetworks/decoder.model` | Model to save or load |
| `--epochs N` | `20000` | Training passes through all 16 inputs |
| `--rate R` | `0.3` | Positive, finite learning rate |
| `--architecture LIST` | `4,12,16` | Comma-separated layer sizes, without spaces |
| `--report N` | `500` | Report loss every N epochs |
| `--no-colour` | Off | Disable decoder colours and screen clearing |

The architecture must start with 4, end with 16, and have at least one
non-empty hidden layer. For example:

```sh
./builds/neural_network decoder train --architecture 4,16,12,16 --epochs 25000 --model neuralNetworks/decoder-deep.model
```

## Section 4: Fashion-MNIST

The dataset is **not included**. Use the training and test CSV files supplied
with the lab. Each file must begin with a header such as
`label,pixel1,...,pixel784`. Every following non-empty row must contain one
label (0–9) and 784 pixel integers (0–255). Pixels are normalised by dividing
by 255; targets are one-hot vectors. Rows are streamed from disk each epoch
in their original order.

Place the files in the project directory, `data/`, `datasets/`, or
`Optional Work/`, or provide their paths explicitly. Relative paths are
resolved from the terminal's current working directory. Quote paths with spaces.

```sh
./builds/neural_network fashion train --train-csv data/fashion-mnist_train.csv --model neuralNetworks/fashion.model --epochs 3
./builds/neural_network fashion test --test-csv data/fashion-mnist_test.csv --model neuralNetworks/fashion.model --preview
./builds/neural_network fashion predict 0 --test-csv data/fashion-mnist_test.csv --model neuralNetworks/fashion.model
./builds/neural_network fashion info --model neuralNetworks/fashion.model
```

`predict INDEX` uses a **zero-based data-row index**, excluding the header and
blank lines. It renders the 28×28 image as terminal characters and prints all
10 output scores. `test` prints accuracy and a confusion matrix whose rows
are actual classes and columns are predicted classes.

| Option | Default | Meaning |
| --- | --- | --- |
| `--train-csv FILE` | `fashion-mnist_train.csv` | Training CSV |
| `--test-csv FILE` | `fashion-mnist_test.csv` | Test or prediction CSV |
| `--model FILE` | `neuralNetworks/fashion.model` | Model to save or load |
| `--epochs N` | `3` | Passes through selected training rows |
| `--rate R` | `0.05` | Positive, finite learning rate |
| `--architecture LIST` | `784,64,10` | Layer sizes; at least one hidden layer |
| `--limit N` | `0` | First N samples for train/test; 0 uses all |
| `--report N` | `1000` | Report loss every N training samples |
| `--preview` | Off | Render the first image during testing |

Fashion-MNIST requires 784 inputs and 10 outputs. For a quick pipeline check:

```sh
./builds/neural_network fashion train --limit 100 --epochs 1 --model neuralNetworks/quick.model
./builds/neural_network fashion test --limit 20 --model neuralNetworks/quick.model
```

To create an untrained model, or try another architecture for section 4.2:

```sh
./builds/neural_network fashion init --architecture 784,128,10 --model neuralNetworks/untrained.model
./builds/neural_network fashion train --architecture 784,128,64,10 --model neuralNetworks/fashion-deep.model
```

Training starts a **new** network; it does not resume a saved model. Use
separate model filenames when comparing architectures. Keep the dataset,
epoch count and sample limit consistent and record the test accuracy yourself.
No section 4.2 experimental results are invented here.

Class indices: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat,
5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.

## Source layout and implementation

| File | Responsibility |
| --- | --- |
| `src/main.cpp` | Single entry point, mode dispatch and main menu |
| `src/decoder.cpp` | Section 3 commands, encoding and training loop |
| `src/fashion.cpp` | Section 4 commands, CSV streaming and image display |
| `src/matrix.cpp`, `includes/matrix.hpp` | Shared dense matrix operations |
| `src/neural_network.cpp`, `includes/neural_network.hpp` | Forward pass, backpropagation and model persistence |
| `includes/commands.hpp` | Section entry-point declarations |
| `includes/cli.hpp` | Shared numeric argument validation |
| `includes/ansi.hpp` | Terminal formatting constants |
| `tests/test_matrix.cpp` | Matrix class tests and their standalone entry point |
| `Makefile`, `tests/Makefile` | CLI and test builds, automatic header dependencies |

The shared implementation uses the readable section 3 code, which implements
the same calculations as section 4. The network uses sigmoid activations,
Xavier uniform weight initialisation, zero biases and per-sample updates.
Reported loss is mean squared error before each update. Backpropagation
propagates the unscaled delta through the old weights before updating them.
Output scores are sigmoid activations, not calibrated class probabilities.

Fixed seeds preserve the original defaults: decoder weights use 5489;
decoder shuffling and Fashion-MNIST weights use 20261023. Numerical results
may vary slightly between standard-library/compiler implementations.

The plain-text model format is unchanged: layer count, layer sizes, learning
rate, then weights and biases for each layer, with 17-digit precision.
Existing models from either section remain readable. Prediction and testing
check the input/output dimensions to reject models from the other task.
Save paths create missing parent directories. Saving replaces an existing file
at that path, so use distinct names to keep previous training runs.

The uploaded `.h` filenames were standardised to `.hpp`, matching the original
source includes. The decoder architecture parser now retains the requested
layers. Numeric CLI inputs reject negative counts, trailing text, non-finite
learning rates and invalid zero values. Command-line errors exit with status 1;
successful commands exit with status 0. Test accuracy is reported as output,
not as a process failure. Fashion-MNIST retains ANSI terminal formatting;
`--no-colour` is a decoder-only option.

## Validation

The combined project was compiled with GCC using C++17, optimisation and
`-Wall -Wextra -Wpedantic`, with no warnings. The default decoder trained and
correctly classified all 16 inputs after saving and loading its model.
The new standalone Matrix test suite passes all 14 test groups. Both Makefiles
were checked, including a clean rebuild, test-only build and header dependency
tracking. Checks also covered custom hidden layers, both menus, invalid CLI arguments,
missing models and rejecting models from the wrong task.

Fashion-MNIST training, testing, image prediction and model initialisation
were exercised using a small synthetic CSV. This verifies the integration,
not real Fashion-MNIST accuracy: the actual dataset was not attached.
