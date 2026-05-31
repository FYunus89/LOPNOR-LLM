#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace lopnor {

struct PropellerInput {
  std::string case_name;
  int blade_count = 0;
  double tip_radius_m = 0.0;
  double hub_radius_m = 0.0;
  double blade_pitch_deg = 0.0;
  std::string blade_geometry_file;
  double rpm = 0.0;
  double shaft_pitch_deg = 0.0;
  double shaft_yaw_deg = 0.0;
  double free_stream_velocity_mps = 0.0;
  double altitude_m = 0.0;
  bool advance_ratio_auto = true;
  double advance_ratio_j = 0.0;
  std::string microphones_file;
};

struct Microphone {
  double x_m = 0.0;
  double y_m = 0.0;
  double z_m = 0.0;
};

struct BladeSection {
  double r_over_R = 0.0;
  double chord_over_R = 0.0;
  double twist_deg = 0.0;
  double sweep_over_R = 0.0;
  std::string airfoil_file;
};

struct Prediction {
  std::vector<double> raw_freq_hz;
  std::vector<double> total_db;
  std::vector<double> broadband_db;
  std::vector<double> tonal_db;
  std::vector<double> one_third_freq_hz;
  std::vector<double> one_third_total_db;
  std::vector<double> line_db;
};

}  // namespace lopnor
