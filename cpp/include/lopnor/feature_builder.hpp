#pragma once

#include <filesystem>
#include <vector>

#include "lopnor/metadata.hpp"
#include "lopnor/types.hpp"

namespace lopnor {

struct ModelInputs {
  std::vector<float> station_feats;  // flattened (batch, S, F)
  std::vector<float> global_feats;   // flattened (batch, G)
  std::vector<float> raw_global;     // flattened (batch, G_raw), before engineered z-score
  std::vector<float> bpf_hz;         // (batch)
  int batch = 0;
};

ModelInputs buildModelInputs(const PropellerInput& input,
                             const std::vector<Microphone>& microphones,
                             const std::vector<BladeSection>& sections,
                             const std::filesystem::path& blade_geometry_path,
                             const Metadata& metadata);

}  // namespace lopnor
