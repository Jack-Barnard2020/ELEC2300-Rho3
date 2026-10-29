/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: cli.hpp
 * File Description: Shared validation for command-line numeric values
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 * ======================================================= */

#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cli {
// Reject signs, suffixes and overflow rather than silently accepting a prefix.
inline std::size_t size(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected a non-negative integer: " + text);
    const auto value = std::stoull(text);
    if (value >= std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Integer is too large: " + text);
    return static_cast<std::size_t>(value);
}
inline double rate(const std::string& text) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument("Learning rate must be positive and finite");
    return value;
}
}
