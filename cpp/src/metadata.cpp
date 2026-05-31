#include "lopnor/metadata.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace lopnor {
namespace {

std::vector<double> asVector(const nlohmann::json& j, const std::string& name) {
  if (!j.is_array()) {
    throw std::runtime_error("Metadata field is not an array: " + name);
  }
  std::vector<double> out;
  out.reserve(j.size());
  for (const auto& item : j) {
    if (item.is_array()) {
      for (const auto& sub : item) {
        out.push_back(sub.get<double>());
      }
    } else {
      out.push_back(item.get<double>());
    }
  }
  return out;
}

const nlohmann::json& require(const nlohmann::json& j, const char* key) {
  if (!j.contains(key)) {
    throw std::runtime_error(std::string("Metadata JSON is missing key: ") + key);
  }
  return j.at(key);
}

}  // namespace

Metadata readMetadata(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open metadata JSON: " + path.string());
  }

  nlohmann::json root;
  in >> root;
  const auto& meta = require(root, "inference_metadata");
  const auto& cfg = require(meta, "model_config");
  const auto& pca = require(meta, "airfoil_pca");
  const auto& norm = require(meta, "normalization");
  const auto& target = require(meta, "target_scaling");
  const auto& freq = require(meta, "frequency_grid");
  const auto& constants = require(meta, "physical_constants");

  Metadata out;
  out.broadband_onnx_file = require(require(root, "onnx_files"), "broadband").get<std::string>();
  out.tonal_onnx_file = require(require(root, "onnx_files"), "tonal").get<std::string>();

  out.config.n_sections = cfg.at("n_sections").get<int>();
  out.config.f_per_section = cfg.at("f_per_section").get<int>();
  out.config.raw_global_dim = cfg.at("raw_global_dim").get<int>();
  out.config.global_dim = cfg.at("global_dim").get<int>();
  out.config.raw_grid_bins = cfg.at("raw_grid_bins").get<int>();
  out.config.target_band_bins = cfg.at("target_band_bins").get<int>();
  out.config.harmonic_count = cfg.at("harmonic_count").get<int>();
  out.config.station_extra_dim = cfg.at("station_extra_dim").get<int>();

  out.n_airfoil_points = pca.at("n_airfoil_points").get<int>();
  out.pca_components = pca.at("n_components_fitted").get<int>();
  out.pca_input_dim = pca.at("input_feature_dim").get<int>();
  out.pca_mean = asVector(pca.at("mean"), "airfoil_pca.mean");
  out.pca_components_matrix = asVector(pca.at("components"), "airfoil_pca.components");

  out.station_geo_mean = asVector(norm.at("station_geo_mean"), "normalization.station_geo_mean");
  out.station_geo_std = asVector(norm.at("station_geo_std"), "normalization.station_geo_std");
  out.global_geo_mean = asVector(norm.at("global_geo_mean"), "normalization.global_geo_mean");
  out.global_geo_std = asVector(norm.at("global_geo_std"), "normalization.global_geo_std");
  out.global_aug_mean = asVector(norm.at("global_aug_mean"), "normalization.global_aug_mean");
  out.global_aug_std = asVector(norm.at("global_aug_std"), "normalization.global_aug_std");

  out.total_raw_mean_db = asVector(target.at("total_raw_mean_db"), "target_scaling.total_raw_mean_db");
  out.total_raw_std_db = asVector(target.at("total_raw_std_db"), "target_scaling.total_raw_std_db");
  out.raw_freq_hz = asVector(freq.at("raw_freq_hz"), "frequency_grid.raw_freq_hz");
  out.raw_low_hz = asVector(freq.at("raw_low_hz"), "frequency_grid.raw_low_hz");
  out.raw_high_hz = asVector(freq.at("raw_high_hz"), "frequency_grid.raw_high_hz");
  out.target_freqs_hz = asVector(freq.at("target_freqs_hz"), "frequency_grid.target_freqs_hz");

  out.reference_distance_m = constants.at("reference_distance_m").get<double>();
  out.power_eps = constants.at("power_eps").get<double>();
  out.tonal_width_floor_bins = constants.at("tonal_width_floor_bins").get<double>();
  out.tonal_width_ceil_bins = constants.at("tonal_width_ceil_bins").get<double>();
  out.tonal_pseudovoigt_mix = constants.at("tonal_pseudovoigt_mix").get<double>();
  return out;
}

}  // namespace lopnor
