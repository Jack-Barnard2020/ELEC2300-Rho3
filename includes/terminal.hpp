/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-29
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: terminal.hpp
 * File Description: Shared terminal theme, screen reset and interactive prompts
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-29)
 * ======================================================= */

#pragma once

#include <string>

// Shared decoder-inspired presentation for every application screen.
namespace ui {
void setEnabled(bool enabled);
bool enabled();
std::string styled(const char* colour, const std::string& text);
void screen(const std::string& section, const std::string& title);
void heading(const std::string& title);
void menuItem(unsigned number, const std::string& label);
void choicePrompt();
std::string prompt(const std::string& section, const std::string& action,
                   const std::string& label, const std::string& explanation,
                   const std::string& current);
void waitForEnter();
void error(const std::string& section, const std::string& message);
}
