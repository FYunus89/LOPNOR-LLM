#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "lopnor/feature_builder.hpp"
#include "lopnor/metadata.hpp"

namespace lopnor {

struct NeuralOutputs {
  std::vector<float> broadband_scaled;  // (batch, N_raw)
  std::vector<float> line_db_ref;       // (batch, H), reference-distance line levels
  std::vector<float> line_width_bins;   // (batch, H)
};

class OnnxRunner {
 public:
  OnnxRunner(const std::filesystem::path& model_dir, const Metadata& metadata);
  ~OnnxRunner();

  NeuralOutputs run(const ModelInputs& inputs, const Metadata& metadata);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lopnor
