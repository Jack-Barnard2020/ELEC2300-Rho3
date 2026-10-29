/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: main.cpp
 * File Description: Single application entry point and section selection
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 *       - Updated executable and model folder paths (2026-10-29)
 *       - Unified terminal theme and in-place screens (2026-10-29)
 * ======================================================= */

#include "commands.hpp"
#include "terminal.hpp"
#include "ansi.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {
const std::string SECTION = "Sections 3 and 4";

void help() {
    ui::screen(SECTION, "Help");
    std::cout << "Usage: ./builds/neural_network <decoder|fashion> <command> [options]\n"
              << "       ./builds/neural_network [menu|help] [--no-colour]\n\n";
    ui::heading("Commands");
    std::cout << "decoder: train, predict <0-15>, test, info, menu, help\n"
              << "fashion: train, predict <index>, test, init, info, menu, help\n\n";
    ui::heading("Terminal options");
    std::cout << "--no-colour  Disable ANSI colours and screen clearing in every mode.\n\n"
              << "Use './builds/neural_network decoder help' or\n"
              << "'./builds/neural_network fashion help' for section-specific options.\n";
}

void menu() {
    std::string choice;
    while (std::cin) {
        ui::screen(SECTION, "Main Menu");
        ui::menuItem(1, "Section 3: 4-bit decoder");
        ui::menuItem(2, "Section 4: Fashion-MNIST");
        ui::menuItem(3, "Help");
        ui::menuItem(4, "Quit");
        ui::choicePrompt();
        if (!std::getline(std::cin, choice)) return;
        char name[] = "neural_network";
        char* args[] = {name, nullptr};
        if (choice == "1") runDecoder(1, args);
        else if (choice == "2") runFashion(1, args);
        else if (choice == "3") { help(); ui::waitForEnter(); }
        else if (choice == "4" || choice == "q") {
            ui::screen(SECTION, "Goodbye");
            std::cout << ui::styled(ansi::GREEN, "Session closed.") << '\n';
            return;
        } else {
            ui::error(SECTION, "Choose a number from 1 to 4.");
            ui::waitForEnter();
        }
    }
}
}

int main(int argc, char** argv) {
    // Apply the terminal policy once so nested menus inherit the same setting.
    // Options taking values keep their following token, even if it looks like a flag.
    std::vector<char*> arguments{argv[0]};
    const std::vector<std::string> valuedOptions = {
        "--model", "--epochs", "--rate", "--report", "--architecture",
        "--train-csv", "--test-csv", "--limit"
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--no-colour") {
            ui::setEnabled(false);
            continue;
        }
        arguments.push_back(argv[index]);
        for (const auto& option : valuedOptions) {
            if (argument == option && index + 1 < argc) {
                arguments.push_back(argv[++index]);
                break;
            }
        }
    }
    const int count = static_cast<int>(arguments.size());
    arguments.push_back(nullptr);
    char** args = arguments.data();
    if (count == 1) { menu(); return 0; }
    const std::string mode = args[1];
    if (mode == "decoder" || mode == "section3") return runDecoder(count - 1, args + 1);
    if (mode == "fashion" || mode == "section4") return runFashion(count - 1, args + 1);
    if (count == 2 && mode == "menu") { menu(); return 0; }
    if (count == 2 && (mode == "help" || mode == "--help" || mode == "-h")) {
        help(); return 0;
    }
    ui::error(SECTION, "Unknown mode or extra arguments. Use './builds/neural_network help'.");
    return 1;
}
