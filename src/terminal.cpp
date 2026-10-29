/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-29
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: terminal.cpp
 * File Description: Shared cyan/yellow theme and render-in-place terminal screens
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-29)
 * ======================================================= */

#include "terminal.hpp"
#include "ansi.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace ui {
namespace {
bool useAnsi = true;
const std::string border(64, '=');
const std::string rule(64, '-');
}

void setEnabled(bool enabled) { useAnsi = enabled; }
bool enabled() { return useAnsi; }

std::string styled(const char* colour, const std::string& text) {
    return useAnsi ? std::string(colour) + text + ansi::RESET : text;
}

void heading(const std::string& title) {
    std::cout << styled(ansi::YELLOW, title) << '\n'
              << styled(ansi::DIM, rule) << '\n';
}

void screen(const std::string& section, const std::string& title) {
    // Clear first, then home the cursor: every new screen replaces the old one.
    // Plain output is an explicit opt-out for redirected logs and simple terminals.
    if (useAnsi) std::cout << ansi::CLEAR_SCREEN << ansi::CURSOR_HOME << ansi::RESET;
    // A previous result may have changed these persistent stream settings.
    std::cout << std::defaultfloat << std::setprecision(6) << std::right
              << styled(ansi::CYAN, border) << '\n'
              << styled(ansi::BOLD, "       ELEC2300 RHo3 Neural Network") << '\n'
              << styled(ansi::CYAN, border) << '\n'
              << styled(ansi::DIM, section) << "\n\n";
    heading(title);
    std::cout << std::flush;
}

void menuItem(unsigned number, const std::string& label) {
    std::cout << styled(ansi::CYAN, std::to_string(number) + ". ") << label << '\n';
}

void choicePrompt() {
    std::cout << '\n' << styled(ansi::CYAN, "Choice: ") << std::flush;
}

std::string prompt(const std::string& section, const std::string& action,
                   const std::string& label, const std::string& explanation,
                   const std::string& current) {
    screen(section, action);
    std::cout << '\n' << styled(ansi::CYAN, label) << '\n'
              << styled(ansi::DIM, explanation) << "\n\n"
              << "Current: " << current << '\n'
              << styled(ansi::CYAN, "Enter value (Enter keeps current): ") << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) throw std::runtime_error("Input ended");
    return line.empty() ? current : line;
}

void waitForEnter() {
    // Keep results visible until acknowledged; the next screen performs the reset.
    if (!std::cin) return;
    std::cout << '\n' << styled(ansi::DIM, "Press Enter to continue...") << std::flush;
    std::string line;
    std::getline(std::cin, line);
}

void error(const std::string& section, const std::string& message) {
    screen(section, "Error");
    std::cerr << styled(ansi::RED, "Error: " + message) << '\n';
}
}
