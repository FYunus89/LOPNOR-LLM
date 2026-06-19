#include "lopnor/input_parser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lopnor {
namespace {

std::string trim(const std::string& s) {
  std::size_t first = 0;
  while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
    ++first;
  }
  std::size_t last = s.size();
  while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
    --last;
  }
  return s.substr(first, last - first);
}

std::string stripComment(const std::string& line) {
  const std::size_t hash = line.find('#');
  if (hash == std::string::npos) {
    return trim(line);
  }
  return trim(line.substr(0, hash));
}

std::vector<std::string> activeLines(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open input file: " + path.string());
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    line = stripComment(line);
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

std::string firstToken(const std::string& line) {
  std::istringstream iss(line);
  std::string token;
  iss >> token;
  if (token.empty()) {
    throw std::runtime_error("Expected a value but found an empty line.");
  }
  return token;
}

double firstDouble(const std::string& line, const std::string& name) {
  try {
    return std::stod(firstToken(line));
  } catch (const std::exception&) {
    throw std::runtime_error("Cannot parse numeric field '" + name + "' from line: " + line);
  }
}

int firstInt(const std::string& line, const std::string& name) {
  try {
    return std::stoi(firstToken(line));
  } catch (const std::exception&) {
    throw std::runtime_error("Cannot parse integer field '" + name + "' from line: " + line);
  }
}

bool isAutoToken(std::string token) {
  std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return token == "auto";
}

}  // namespace

std::filesystem::path resolveRelativeTo(const std::filesystem::path& base_file,
                                        const std::string& child) {
  std::filesystem::path p(child);
  if (p.is_absolute()) {
    return p;
  }
  return base_file.parent_path() / p;
}

PropellerInput readMenuFile(const std::filesystem::path& path) {
  const auto lines = activeLines(path);
  if (lines.size() < 12) {
    throw std::runtime_error(
        "LOPNOR_LLM.i needs at least 12 active value lines: case, blade count, tip radius, "
        "hub radius, blade pitch, geometry file, RPM, shaft pitch, shaft yaw, free-stream "
        "velocity, advance ratio or AUTO, and microphones file.");
  }

  PropellerInput out;
  out.case_name = firstToken(lines[0]);
  out.blade_count = firstInt(lines[1], "blade_count");
  out.tip_radius_m = firstDouble(lines[2], "tip_radius_m");
  out.hub_radius_m = firstDouble(lines[3], "hub_radius_m");
  out.blade_pitch_deg = firstDouble(lines[4], "blade_pitch_deg");
  out.blade_geometry_file = firstToken(lines[5]);
  out.rpm = firstDouble(lines[6], "rpm");
  out.shaft_pitch_deg = firstDouble(lines[7], "shaft_pitch_deg");
  out.shaft_yaw_deg = firstDouble(lines[8], "shaft_yaw_deg");
  out.free_stream_velocity_mps = firstDouble(lines[9], "free_stream_velocity_mps");
  out.altitude_m = 0.0;
  const std::size_t j_index = lines.size() >= 13 ? 11 : 10;
  const std::size_t microphones_index = lines.size() >= 13 ? 12 : 11;
  if (lines.size() >= 13) {
    out.altitude_m = firstDouble(lines[10], "altitude_m");
  }
  const std::string j_token = firstToken(lines[j_index]);
  out.advance_ratio_auto = isAutoToken(j_token);
  if (!out.advance_ratio_auto) {
    out.advance_ratio_j = std::stod(j_token);
  }
  out.microphones_file = firstToken(lines[microphones_index]);

  if (out.advance_ratio_auto) {
    const double n_rev_per_s = out.rpm / 60.0;
    const double diameter_m = 2.0 * out.tip_radius_m;
    const double denom = n_rev_per_s * diameter_m;
    out.advance_ratio_j = denom > 1.0e-12 ? out.free_stream_velocity_mps / denom : 0.0;
  }
  return out;
}

std::vector<Microphone> readMicrophones(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open microphone file: " + path.string());
  }
  int n = 0;
  in >> n;
  if (n <= 0) {
    throw std::runtime_error("Microphone file must start with a positive microphone count.");
  }
  std::vector<Microphone> mics(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (!(in >> mics[static_cast<std::size_t>(i)].x_m >> mics[static_cast<std::size_t>(i)].y_m >>
          mics[static_cast<std::size_t>(i)].z_m)) {
      throw std::runtime_error("Microphone file ended before all x y z rows were read.");
    }
  }
  return mics;
}

std::vector<BladeSection> readBladeGeometry(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open blade geometry file: " + path.string());
  }

  std::vector<BladeSection> sections;
  std::string line;
  while (std::getline(in, line)) {
    line = stripComment(line);
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    BladeSection s;
    if (!(iss >> s.r_over_R >> s.chord_over_R >> s.twist_deg >> s.sweep_over_R >> s.airfoil_file)) {
      throw std::runtime_error("Cannot parse blade geometry row: " + line);
    }
    sections.push_back(s);
  }
  if (sections.empty()) {
    throw std::runtime_error("Blade geometry file contains no usable sections: " + path.string());
  }
  return sections;
}

}  // namespace lopnor
