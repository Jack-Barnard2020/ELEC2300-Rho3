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
 * ======================================================= */

#include "commands.hpp"
#include <iostream>
#include <string>

namespace {
void help() {
    std::cout << "ELEC2300 RHo3 - Sections 3 and 4\n\n"
              << "Usage: ./builds/neural_network <decoder|fashion> <command> [options]\n"
              << "       ./builds/neural_network [menu|help]\n\n"
              << "decoder: train, predict <0-15>, test, info, menu, help\n"
              << "fashion: train, predict <index>, test, init, info, menu, help\n\n"
              << "Use './builds/neural_network decoder help' or './builds/neural_network fashion help'\n"
              << "for section-specific options. No arguments opens the menu.\n";
}
void menu() {
    std::string choice;
    while (std::cin) {
        std::cout << "\nELEC2300 RHo3\n1. Section 3: 4-bit decoder\n"
                  << "2. Section 4: Fashion-MNIST\n3. Help\n4. Quit\nChoice: ";
        if (!std::getline(std::cin, choice)) return;
        // Reuse the same section entry points for interactive and scripted use.
        char name[] = "neural_network";
        char* args[] = {name, nullptr};
        if (choice == "1") runDecoder(1, args);
        else if (choice == "2") runFashion(1, args);
        else if (choice == "3") help();
        else if (choice == "4" || choice == "q") return;
        else std::cerr << "Choose 1, 2, 3 or 4.\n";
    }
}
}
int main(int argc, char** argv) {
    if (argc == 1) { menu(); return 0; }
    const std::string mode = argv[1];
    // Removing the mode leaves each section's existing action parser intact.
    if (mode == "decoder" || mode == "section3") return runDecoder(argc - 1, argv + 1);
    if (mode == "fashion" || mode == "section4") return runFashion(argc - 1, argv + 1);
    if (argc == 2 && mode == "menu") { menu(); return 0; }
    if (argc == 2 && (mode == "help" || mode == "--help" || mode == "-h")) {
        help(); return 0;
    }
    std::cerr << "Unknown mode or extra arguments. Use './builds/neural_network help'.\n";
    return 1;
}
