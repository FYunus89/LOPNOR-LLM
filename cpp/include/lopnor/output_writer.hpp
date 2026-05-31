#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "lopnor/types.hpp"

namespace lopnor {

void writeSpectrumCsv(const std::filesystem::path& path, const Prediction& prediction);

void writeBandCsv(const std::filesystem::path& path, const Prediction& prediction);

void writePreviewWav(const std::filesystem::path& path,
                     const Prediction& prediction,
                     double seconds,
                     int sample_rate,
                     unsigned int seed);

std::string micStem(int index_zero_based);

}  // namespace lopnor
