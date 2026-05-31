#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "lopnor/types.hpp"

namespace lopnor {

PropellerInput readMenuFile(const std::filesystem::path& path);

std::vector<Microphone> readMicrophones(const std::filesystem::path& path);

std::vector<BladeSection> readBladeGeometry(const std::filesystem::path& path);

std::filesystem::path resolveRelativeTo(const std::filesystem::path& base_file,
                                        const std::string& child);

}  // namespace lopnor
