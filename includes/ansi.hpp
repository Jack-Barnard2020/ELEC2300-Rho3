/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: ansi.hpp
 * File Description: ANSI escape codes for terminal formatting
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 * ======================================================= */

#pragma once

namespace ansi {
constexpr const char* RESET        = "\033[0m";
constexpr const char* BOLD         = "\033[1m";
constexpr const char* DIM          = "\033[2m";
constexpr const char* RED          = "\033[31m";
constexpr const char* GREEN        = "\033[32m";
constexpr const char* YELLOW       = "\033[33m";
constexpr const char* CYAN         = "\033[36m";
constexpr const char* CLEAR_SCREEN = "\033[2J";
constexpr const char* CURSOR_HOME  = "\033[H";
}
