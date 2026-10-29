/* ==================== ELEC2300 RHo3 ====================
 * Author: Jack Barnard
 * Date: 2026-10-23
 * Project: RHo3
 * Project Description: Neural networks for a 4-bit decoder and Fashion-MNIST
 * File: commands.hpp
 * File Description: Entry points for the two CLI modes
 * Change Log:
 *       - Initial version (Jack Barnard, 2026-10-23)
 *       - Combined sections 3 and 4 into one CLI (2026-10-29)
 * ======================================================= */

#pragma once

// Each section accepts its action in argv[1]; main removes the section name.
int runDecoder(int argc, char** argv);
int runFashion(int argc, char** argv);
