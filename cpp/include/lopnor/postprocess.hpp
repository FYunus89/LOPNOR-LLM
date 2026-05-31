#pragma once

#include <vector>

#include "lopnor/feature_builder.hpp"
#include "lopnor/metadata.hpp"
#include "lopnor/onnx_runner.hpp"
#include "lopnor/types.hpp"

namespace lopnor {

std::vector<Prediction> postprocessPredictions(const NeuralOutputs& neural,
                                               const ModelInputs& inputs,
                                               const Metadata& metadata);

}  // namespace lopnor
