#include "lopnor/onnx_runner.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

namespace lopnor {
namespace {

std::filesystem::path joinModelPath(const std::filesystem::path& model_dir, const std::string& file) {
  std::filesystem::path p(file);
  if (p.is_absolute()) {
    return p;
  }
  return model_dir / p;
}

std::vector<float> tensorToVector(const Ort::Value& value) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  const std::size_t count = info.GetElementCount();
  const float* ptr = value.GetTensorData<float>();
  return std::vector<float>(ptr, ptr + count);
}

}  // namespace

struct OnnxRunner::Impl {
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session broadband_session;
  Ort::Session tonal_session;

  Impl(const std::filesystem::path& model_dir, const Metadata& metadata)
      : env(ORT_LOGGING_LEVEL_WARNING, "lopnor"),
        session_options(),
        broadband_session(nullptr),
        tonal_session(nullptr) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    const auto broadband_path = joinModelPath(model_dir, metadata.broadband_onnx_file);
    const auto tonal_path = joinModelPath(model_dir, metadata.tonal_onnx_file);
    broadband_session = Ort::Session(env, broadband_path.c_str(), session_options);
    tonal_session = Ort::Session(env, tonal_path.c_str(), session_options);
  }
};

OnnxRunner::OnnxRunner(const std::filesystem::path& model_dir, const Metadata& metadata)
    : impl_(std::make_unique<Impl>(model_dir, metadata)) {}

OnnxRunner::~OnnxRunner() = default;

NeuralOutputs OnnxRunner::run(const ModelInputs& inputs, const Metadata& metadata) {
  if (inputs.batch <= 0) {
    throw std::runtime_error("Cannot run ONNX model with an empty batch.");
  }

  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 3> station_shape = {
      static_cast<int64_t>(inputs.batch),
      static_cast<int64_t>(metadata.config.n_sections),
      static_cast<int64_t>(metadata.config.f_per_section),
  };
  const std::array<int64_t, 2> global_shape = {
      static_cast<int64_t>(inputs.batch),
      static_cast<int64_t>(metadata.config.global_dim),
  };
  const std::array<int64_t, 1> bpf_shape = {static_cast<int64_t>(inputs.batch)};

  auto station_tensor = Ort::Value::CreateTensor<float>(
      memory,
      const_cast<float*>(inputs.station_feats.data()),
      inputs.station_feats.size(),
      station_shape.data(),
      station_shape.size());
  auto global_tensor = Ort::Value::CreateTensor<float>(
      memory,
      const_cast<float*>(inputs.global_feats.data()),
      inputs.global_feats.size(),
      global_shape.data(),
      global_shape.size());

  const char* bb_inputs[] = {"station_feats", "global_feats"};
  const char* bb_outputs[] = {"broadband_scaled", "broadband_token_weights"};
  std::array<Ort::Value, 2> bb_input_values = {std::move(station_tensor), std::move(global_tensor)};
  auto bb_values = impl_->broadband_session.Run(
      Ort::RunOptions{nullptr},
      bb_inputs,
      bb_input_values.data(),
      bb_input_values.size(),
      bb_outputs,
      2);

  auto station_tensor_tonal = Ort::Value::CreateTensor<float>(
      memory,
      const_cast<float*>(inputs.station_feats.data()),
      inputs.station_feats.size(),
      station_shape.data(),
      station_shape.size());
  auto global_tensor_tonal = Ort::Value::CreateTensor<float>(
      memory,
      const_cast<float*>(inputs.global_feats.data()),
      inputs.global_feats.size(),
      global_shape.data(),
      global_shape.size());
  auto bpf_tensor = Ort::Value::CreateTensor<float>(
      memory,
      const_cast<float*>(inputs.bpf_hz.data()),
      inputs.bpf_hz.size(),
      bpf_shape.data(),
      bpf_shape.size());

  const char* tonal_inputs[] = {"station_feats", "global_feats", "bpf_hz"};
  const char* tonal_outputs[] = {
      "line_db_ref",
      "tonal_token_weights",
      "tail_decay",
      "line_width_bins",
      "tail_curvature",
      "high_order_drop",
  };
  std::array<Ort::Value, 3> tonal_input_values = {
      std::move(station_tensor_tonal),
      std::move(global_tensor_tonal),
      std::move(bpf_tensor),
  };
  auto tonal_values = impl_->tonal_session.Run(
      Ort::RunOptions{nullptr},
      tonal_inputs,
      tonal_input_values.data(),
      tonal_input_values.size(),
      tonal_outputs,
      6);

  NeuralOutputs out;
  out.broadband_scaled = tensorToVector(bb_values[0]);
  out.line_db_ref = tensorToVector(tonal_values[0]);
  out.line_width_bins = tensorToVector(tonal_values[3]);
  return out;
}

}  // namespace lopnor
