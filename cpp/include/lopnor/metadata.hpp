#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace lopnor {

struct ModelConfig {
  int n_sections = 0;
  int f_per_section = 0;
  int raw_global_dim = 0;
  int global_dim = 0;
  int raw_grid_bins = 0;
  int target_band_bins = 0;
  int harmonic_count = 0;
  int station_extra_dim = 0;
};

struct Metadata {
  ModelConfig config;
  std::string broadband_onnx_file;
  std::string tonal_onnx_file;

  int n_airfoil_points = 180;
  int pca_components = 10;
  int pca_input_dim = 360;
  std::vector<double> pca_mean;
  std::vector<double> pca_components_matrix;

  std::vector<double> station_geo_mean;
  std::vector<double> station_geo_std;
  std::vector<double> global_geo_mean;
  std::vector<double> global_geo_std;
  std::vector<double> global_aug_mean;
  std::vector<double> global_aug_std;

  std::vector<double> total_raw_mean_db;
  std::vector<double> total_raw_std_db;
  std::vector<double> raw_freq_hz;
  std::vector<double> raw_low_hz;
  std::vector<double> raw_high_hz;
  std::vector<double> target_freqs_hz;

  double reference_distance_m = 1.0;
  double power_eps = 1.0e-12;
  double tonal_width_floor_bins = 0.45;
  double tonal_width_ceil_bins = 1.10;
  double tonal_pseudovoigt_mix = 0.06;
};

Metadata readMetadata(const std::filesystem::path& path);

}  // namespace lopnor
