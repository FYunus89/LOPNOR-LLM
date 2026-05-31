#include "lopnor/feature_builder.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace lopnor {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kDistEps = 1.0e-6;

double clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

std::vector<double> linspace(double lo, double hi, int n) {
  std::vector<double> x(static_cast<std::size_t>(n));
  if (n <= 1) {
    if (n == 1) {
      x[0] = lo;
    }
    return x;
  }
  for (int i = 0; i < n; ++i) {
    x[static_cast<std::size_t>(i)] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
  }
  return x;
}

double trapz(const std::vector<double>& y, const std::vector<double>& x, int n) {
  double sum = 0.0;
  for (int i = 1; i < n; ++i) {
    sum += 0.5 * (y[static_cast<std::size_t>(i)] + y[static_cast<std::size_t>(i - 1)]) *
           (x[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i - 1)]);
  }
  return sum;
}

double interp1(const std::vector<double>& x, const std::vector<double>& y, double q) {
  if (x.empty() || y.empty() || x.size() != y.size()) {
    throw std::runtime_error("Invalid interpolation table.");
  }
  if (q <= x.front()) {
    return y.front();
  }
  if (q >= x.back()) {
    return y.back();
  }
  const auto it = std::lower_bound(x.begin(), x.end(), q);
  const std::size_t hi = static_cast<std::size_t>(std::distance(x.begin(), it));
  const std::size_t lo = hi - 1;
  const double t = (q - x[lo]) / std::max(x[hi] - x[lo], 1.0e-12);
  return y[lo] + t * (y[hi] - y[lo]);
}

double slopeLastPoints(const std::vector<double>& x, const std::vector<double>& y, int count) {
  const int n = static_cast<int>(x.size());
  const int first = std::max(0, n - count);
  const int m = n - first;
  if (m < 2) {
    return 0.0;
  }
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  for (int i = first; i < n; ++i) {
    sx += x[static_cast<std::size_t>(i)];
    sy += y[static_cast<std::size_t>(i)];
    sxx += x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    sxy += x[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
  }
  const double denom = static_cast<double>(m) * sxx - sx * sx;
  if (std::abs(denom) < 1.0e-12) {
    return 0.0;
  }
  return (static_cast<double>(m) * sxy - sx * sy) / denom;
}

struct AirfoilSurfaces {
  std::vector<double> x;
  std::vector<double> upper;
  std::vector<double> lower;
};

std::vector<std::pair<double, double>> uniqueSortedPairs(std::vector<std::pair<double, double>> pairs) {
  std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<std::pair<double, double>> out;
  for (const auto& p : pairs) {
    if (out.empty() || std::abs(p.first - out.back().first) > 1.0e-12) {
      out.push_back(p);
    }
  }
  return out;
}

AirfoilSurfaces readAirfoilSurfaces(const std::filesystem::path& path, int n_points) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open airfoil file: " + path.string());
  }

  std::vector<double> x_raw;
  std::vector<double> y_raw;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    double x = 0.0;
    double y = 0.0;
    if (iss >> x >> y) {
      x_raw.push_back(x);
      y_raw.push_back(y);
    }
  }
  if (x_raw.size() < 3) {
    throw std::runtime_error("Airfoil file has too few numeric coordinate rows: " + path.string());
  }

  const auto [xmin_it, xmax_it] = std::minmax_element(x_raw.begin(), x_raw.end());
  const double xmin = *xmin_it;
  const double xmax = *xmax_it;
  const double dx = std::max(xmax - xmin, 1.0e-12);
  std::vector<double> x_norm(x_raw.size());
  for (std::size_t i = 0; i < x_raw.size(); ++i) {
    x_norm[i] = (x_raw[i] - xmin) / dx;
  }
  const auto le_it = std::min_element(x_norm.begin(), x_norm.end());
  const std::size_t le = static_cast<std::size_t>(std::distance(x_norm.begin(), le_it));

  std::vector<std::pair<double, double>> upper_pairs;
  for (std::size_t k = 0; k <= le; ++k) {
    const std::size_t i = le - k;
    upper_pairs.emplace_back(x_norm[i], y_raw[i]);
  }
  std::vector<std::pair<double, double>> lower_pairs;
  for (std::size_t i = le; i < x_norm.size(); ++i) {
    lower_pairs.emplace_back(x_norm[i], y_raw[i]);
  }
  upper_pairs = uniqueSortedPairs(std::move(upper_pairs));
  lower_pairs = uniqueSortedPairs(std::move(lower_pairs));

  std::vector<double> xu;
  std::vector<double> yu;
  std::vector<double> xl;
  std::vector<double> yl;
  for (const auto& p : upper_pairs) {
    xu.push_back(p.first);
    yu.push_back(p.second);
  }
  for (const auto& p : lower_pairs) {
    xl.push_back(p.first);
    yl.push_back(p.second);
  }

  AirfoilSurfaces out;
  out.x = linspace(0.0, 1.0, n_points);
  out.upper.resize(static_cast<std::size_t>(n_points));
  out.lower.resize(static_cast<std::size_t>(n_points));
  for (int i = 0; i < n_points; ++i) {
    const double q = out.x[static_cast<std::size_t>(i)];
    out.upper[static_cast<std::size_t>(i)] = interp1(xu, yu, q);
    out.lower[static_cast<std::size_t>(i)] = interp1(xl, yl, q);
  }
  return out;
}

std::vector<double> computeBaseStationFeatures(const std::vector<BladeSection>& sections,
                                               const std::filesystem::path& blade_geometry_path,
                                               const Metadata& metadata) {
  const int S = metadata.config.n_sections;
  const int base_f = metadata.config.f_per_section - metadata.config.station_extra_dim;
  const int n_pca = std::min(metadata.pca_components, 10);
  const int n_points = metadata.n_airfoil_points;
  if (base_f < n_pca + 3) {
    throw std::runtime_error("Metadata station feature dimensions are inconsistent.");
  }
  std::vector<double> station(static_cast<std::size_t>(S * base_f), 0.0);
  const int n_use = std::min<int>(S, static_cast<int>(sections.size()));
  for (int i = 0; i < n_use; ++i) {
    const auto airfoil_path = blade_geometry_path.parent_path() / sections[static_cast<std::size_t>(i)].airfoil_file;
    const auto af = readAirfoilSurfaces(airfoil_path, n_points);
    std::vector<double> combined(static_cast<std::size_t>(2 * n_points), 0.0);
    for (int j = 0; j < n_points; ++j) {
      const double mean = 0.5 * (af.upper[static_cast<std::size_t>(j)] + af.lower[static_cast<std::size_t>(j)]);
      const double thick = af.upper[static_cast<std::size_t>(j)] - af.lower[static_cast<std::size_t>(j)];
      combined[static_cast<std::size_t>(j)] = mean;
      combined[static_cast<std::size_t>(n_points + j)] = thick;
    }
    for (int c = 0; c < n_pca; ++c) {
      double value = 0.0;
      for (int j = 0; j < metadata.pca_input_dim; ++j) {
        const double centered = combined[static_cast<std::size_t>(j)] - metadata.pca_mean[static_cast<std::size_t>(j)];
        value += centered * metadata.pca_components_matrix[static_cast<std::size_t>(c * metadata.pca_input_dim + j)];
      }
      station[static_cast<std::size_t>(i * base_f + c)] = value;
    }
    station[static_cast<std::size_t>(i * base_f + n_pca + 0)] = sections[static_cast<std::size_t>(i)].chord_over_R;
    station[static_cast<std::size_t>(i * base_f + n_pca + 1)] = sections[static_cast<std::size_t>(i)].twist_deg;
    station[static_cast<std::size_t>(i * base_f + n_pca + 2)] = sections[static_cast<std::size_t>(i)].sweep_over_R;
  }
  return station;
}

void computeExplicitGeometry(const std::vector<BladeSection>& sections,
                             const std::filesystem::path& blade_geometry_path,
                             const Metadata& metadata,
                             std::vector<double>& station_extra,
                             std::vector<double>& global_extra) {
  const int S = metadata.config.n_sections;
  const int n_points = metadata.n_airfoil_points;
  station_extra.assign(static_cast<std::size_t>(S * 8), 0.0);
  global_extra.assign(19, 0.0);
  const int n_use = std::min<int>(S, static_cast<int>(sections.size()));
  if (n_use <= 0) {
    return;
  }

  std::vector<double> r(n_use), chord(n_use), twist(n_use), sweep(n_use);
  for (int i = 0; i < n_use; ++i) {
    const auto& sec = sections[static_cast<std::size_t>(i)];
    r[static_cast<std::size_t>(i)] = sec.r_over_R;
    chord[static_cast<std::size_t>(i)] = sec.chord_over_R;
    twist[static_cast<std::size_t>(i)] = sec.twist_deg;
    sweep[static_cast<std::size_t>(i)] = sec.sweep_over_R;
    const auto af = readAirfoilSurfaces(blade_geometry_path.parent_path() / sec.airfoil_file, n_points);
    std::vector<double> thickness(static_cast<std::size_t>(n_points));
    std::vector<double> camber(static_cast<std::size_t>(n_points));
    for (int j = 0; j < n_points; ++j) {
      thickness[static_cast<std::size_t>(j)] =
          std::max(af.upper[static_cast<std::size_t>(j)] - af.lower[static_cast<std::size_t>(j)], 0.0);
      camber[static_cast<std::size_t>(j)] =
          0.5 * (af.upper[static_cast<std::size_t>(j)] + af.lower[static_cast<std::size_t>(j)]);
    }
    const double area_norm = trapz(thickness, af.x, n_points);
    const double section_area = chord[static_cast<std::size_t>(i)] * chord[static_cast<std::size_t>(i)] * area_norm;
    const double te95 = interp1(af.x, thickness, 0.95);
    const double te99 = interp1(af.x, thickness, 0.99);
    const double cam95 = interp1(af.x, camber, 0.95);
    const auto max_it = std::max_element(thickness.begin(), thickness.end());
    const int imax = static_cast<int>(std::distance(thickness.begin(), max_it));
    const double max_thick = *max_it;
    const double x_tmax = af.x[static_cast<std::size_t>(imax)];
    const double slope_up = slopeLastPoints(af.x, af.upper, 10);
    const double slope_lo = slopeLastPoints(af.x, af.lower, 10);
    const double te_wedge = std::abs(slope_up - slope_lo);
    const double vals[8] = {r[static_cast<std::size_t>(i)], te95, te99, te_wedge, max_thick,
                            x_tmax + cam95, area_norm, section_area};
    for (int k = 0; k < 8; ++k) {
      station_extra[static_cast<std::size_t>(i * 8 + k)] = vals[k];
    }
  }

  std::vector<double> tip_w(n_use);
  double tip_sum = 0.0;
  for (int i = 0; i < n_use; ++i) {
    const double raw = std::pow(clamp((r[static_cast<std::size_t>(i)] - 0.65) / 0.35, 0.0, 1.0), 2.0) + 1.0e-3;
    tip_w[static_cast<std::size_t>(i)] = raw;
    tip_sum += raw;
  }
  for (double& v : tip_w) {
    v /= std::max(tip_sum, 1.0e-12);
  }

  auto station_col = [&](int i, int col) { return station_extra[static_cast<std::size_t>(i * 8 + col)]; };
  std::vector<double> section_area(n_use), area_norm(n_use), te95(n_use), te99(n_use), wedge(n_use), max_thick(n_use);
  std::vector<double> chord_te(n_use);
  for (int i = 0; i < n_use; ++i) {
    te95[static_cast<std::size_t>(i)] = station_col(i, 1);
    te99[static_cast<std::size_t>(i)] = station_col(i, 2);
    wedge[static_cast<std::size_t>(i)] = station_col(i, 3);
    max_thick[static_cast<std::size_t>(i)] = station_col(i, 4);
    area_norm[static_cast<std::size_t>(i)] = station_col(i, 6);
    section_area[static_cast<std::size_t>(i)] = station_col(i, 7);
    chord_te[static_cast<std::size_t>(i)] = chord[static_cast<std::size_t>(i)] * te99[static_cast<std::size_t>(i)];
  }

  auto tip_mean = [&](const std::vector<double>& v) {
    double sum = 0.0;
    for (int i = 0; i < n_use; ++i) {
      sum += tip_w[static_cast<std::size_t>(i)] * v[static_cast<std::size_t>(i)];
    }
    return sum;
  };
  std::vector<double> chord_r3(n_use), area_r2(n_use);
  for (int i = 0; i < n_use; ++i) {
    chord_r3[static_cast<std::size_t>(i)] = chord[static_cast<std::size_t>(i)] * std::pow(r[static_cast<std::size_t>(i)], 3.0);
    area_r2[static_cast<std::size_t>(i)] = section_area[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)];
  }
  global_extra = {
      tip_mean(te95),
      tip_mean(te99),
      tip_mean(wedge),
      tip_mean(max_thick),
      tip_mean(chord_te),
      tip_mean(area_norm),
      tip_mean(section_area),
      trapz(section_area, r, n_use),
      trapz(area_r2, r, n_use),
      interp1(r, section_area, 0.75),
      interp1(r, section_area, 0.90),
      trapz(chord, r, n_use),
      trapz(chord_r3, r, n_use),
      interp1(r, chord, 0.75),
      interp1(r, chord, 0.90),
      interp1(r, twist, 0.75),
      interp1(r, twist, 0.90),
      interp1(r, twist, 0.75) - interp1(r, twist, 0.90),
      tip_mean(sweep),
  };
}

std::vector<double> makeRawGlobal(const PropellerInput& input, const Microphone& mic, const Metadata& metadata) {
  std::vector<double> g(static_cast<std::size_t>(metadata.config.raw_global_dim), 0.0);
  if (g.size() < 13) {
    throw std::runtime_error("Model expects raw_global_dim >= 13.");
  }
  g[0] = static_cast<double>(input.blade_count) / 10.0;
  g[1] = input.tip_radius_m;
  g[2] = input.hub_radius_m;
  g[3] = input.free_stream_velocity_mps / 60.0;
  g[4] = input.rpm / 8000.0;
  g[5] = input.blade_pitch_deg / 15.0;
  g[6] = input.shaft_pitch_deg / 15.0;
  g[7] = input.shaft_yaw_deg / 100.0;
  g[8] = input.altitude_m;
  g[9] = input.advance_ratio_j;
  g[g.size() - 3] = mic.x_m;
  g[g.size() - 2] = mic.y_m;
  g[g.size() - 1] = mic.z_m;
  return g;
}

double receiverDistance(const std::vector<double>& g) {
  const double x = g[g.size() - 3];
  const double y = g[g.size() - 2];
  const double z = g[g.size() - 1];
  return std::max(std::sqrt(x * x + y * y + z * z), kDistEps);
}

std::vector<double> receiverFeaturesWithAngles(std::vector<double> g) {
  const double pitch_deg_raw = g[6] * 15.0;
  const double yaw_deg = g[7] * 100.0;
  double pitch_wrapped = std::fmod(pitch_deg_raw + 180.0, 360.0);
  if (pitch_wrapped < 0.0) {
    pitch_wrapped += 360.0;
  }
  pitch_wrapped -= 180.0;
  const bool flip = std::abs(pitch_wrapped) > 90.0;
  double pitch_tilt = pitch_wrapped;
  if (flip) {
    pitch_tilt -= (pitch_tilt >= 0.0 ? 1.0 : -1.0) * 180.0;
  }
  g[6] = pitch_tilt / 15.0;

  const double r = receiverDistance(g);
  const double tip_radius = std::max(g[1], kDistEps);
  const double diameter = std::max(2.0 * tip_radius, kDistEps);
  const double r_over_d = r / diameter;
  const double xdir = g[g.size() - 3] / r;
  const double ydir = g[g.size() - 2] / r;
  const double zdir = g[g.size() - 1] / r;

  std::vector<double> out = g;
  out.push_back(r);
  out.push_back(std::log10(r));
  out.push_back(r_over_d);
  out.push_back(std::log10(std::max(r_over_d, kDistEps)));
  out.push_back(xdir);
  out.push_back(ydir);
  out.push_back(zdir);

  const double pitch_rad = pitch_tilt * kPi / 180.0;
  const double yaw_rad = yaw_deg * kPi / 180.0;
  const double cp = std::cos(pitch_rad);
  const double sp = std::sin(pitch_rad);
  const double cy = std::cos(yaw_rad);
  const double sy = std::sin(yaw_rad);
  const double ax[3] = {cp * cy, cp * sy, sp};
  const double ay[3] = {-sy, cy, 0.0};
  const double az[3] = {-sp * cy, -sp * sy, cp};
  const double obs[3] = {xdir, ydir, zdir};
  const double obs_axial = obs[0] * ax[0] + obs[1] * ax[1] + obs[2] * ax[2];
  const double obs_lateral = obs[0] * ay[0] + obs[1] * ay[1] + obs[2] * ay[2];
  const double obs_vertical = obs[0] * az[0] + obs[1] * az[1] + obs[2] * az[2];
  const double obs_transverse = std::sqrt(std::max(obs_lateral * obs_lateral + obs_vertical * obs_vertical, 0.0));
  const double rotor_theta = std::acos(clamp(obs_axial, -1.0, 1.0)) * 180.0 / kPi / 90.0;
  const double pitch_plane = std::atan2(obs_vertical, obs_axial) * 180.0 / kPi / 90.0;
  const double lateral_plane = std::atan2(obs_lateral, obs_axial) * 180.0 / kPi / 90.0;
  out.push_back(pitch_tilt);
  out.push_back(flip ? 1.0 : 0.0);
  out.push_back(obs_axial);
  out.push_back(obs_lateral);
  out.push_back(obs_vertical);
  out.push_back(obs_transverse);
  out.push_back(rotor_theta);
  out.push_back(pitch_plane);
  out.push_back(lateral_plane);
  return out;
}

std::vector<double> operatingFeatures(const std::vector<double>& g) {
  const double nb = std::max(g[0] * 10.0, 1.0e-3);
  const double r_tip = std::max(g[1], 1.0e-4);
  const double v_inf = g[3] * 60.0;
  const double rpm = std::max(g[4] * 8000.0, 1.0e-3);
  const double blade_pitch_deg = g[5] * 15.0;
  const double shaft_pitch_deg = g[6] * 15.0;
  const double altitude_m = g[8];
  const double j = g[9];
  const double bpf_hz = nb * rpm / 60.0;
  const double omega = rpm * (2.0 * kPi / 60.0);
  const double tip_speed = omega * r_tip;
  const double helical_tip_speed = std::sqrt(std::max(tip_speed * tip_speed + v_inf * v_inf, 1.0e-9));
  const double temp_k = std::max(288.15 - 0.0065 * clamp(altitude_m, 0.0, 11000.0), 216.65);
  const double sound_speed = std::sqrt(std::max(1.4 * 287.05 * temp_k, 1.0e-6));
  const double tip_mach = helical_tip_speed / sound_speed;
  const double axial_mach = std::abs(v_inf) / sound_speed;
  const double shaft_pitch_rad = shaft_pitch_deg * kPi / 180.0;
  const double blade_pitch_rad = blade_pitch_deg * kPi / 180.0;
  const double axial_ratio = (v_inf * std::cos(shaft_pitch_rad)) / std::max(tip_speed, 1.0e-6);
  const double crossflow_ratio = (v_inf * std::sin(shaft_pitch_rad)) / std::max(tip_speed, 1.0e-6);
  const double phi75 = std::atan2(v_inf * std::cos(shaft_pitch_rad), std::max(omega * r_tip * 0.75, 1.0e-6));
  const double phi90 = std::atan2(v_inf * std::cos(shaft_pitch_rad), std::max(omega * r_tip * 0.90, 1.0e-6));
  const double beta75 = blade_pitch_rad - phi75;
  const double beta90 = blade_pitch_rad - phi90;
  const double loading = tip_mach * clamp(1.0 - j, -2.0, 2.0);
  return {
      std::log10(std::max(rpm / 1000.0, 1.0e-6)),
      std::log10(std::max(bpf_hz / 100.0, 1.0e-6)),
      std::log10(std::max(tip_speed, 1.0e-3)),
      std::log10(std::max(helical_tip_speed, 1.0e-3)),
      tip_mach,
      axial_mach,
      loading,
      blade_pitch_rad,
      shaft_pitch_rad,
      axial_ratio,
      crossflow_ratio,
      phi75,
      phi90,
      beta75,
      beta90,
      tip_mach * beta75,
      tip_mach * beta90,
      j,
  };
}

std::vector<double> directivityBasis(const std::vector<double>& g) {
  const std::vector<double> rf = receiverFeaturesWithAngles(g);
  const int n = static_cast<int>(rf.size());
  const double obs_axial = rf[static_cast<std::size_t>(n - 7)];
  const double obs_lateral = rf[static_cast<std::size_t>(n - 6)];
  const double obs_vertical = rf[static_cast<std::size_t>(n - 5)];
  const double obs_transverse = rf[static_cast<std::size_t>(n - 4)];
  const double rotor_theta = rf[static_cast<std::size_t>(n - 3)];
  const double pitch_plane = rf[static_cast<std::size_t>(n - 2)];
  const double lateral_plane = rf[static_cast<std::size_t>(n - 1)];
  return {
      obs_axial,
      obs_lateral,
      obs_vertical,
      obs_transverse,
      obs_axial * obs_vertical,
      obs_axial * obs_lateral,
      obs_vertical * obs_lateral,
      obs_axial * obs_axial - (1.0 / 3.0),
      obs_vertical * obs_vertical - obs_lateral * obs_lateral,
      rotor_theta,
      pitch_plane,
      lateral_plane,
  };
}

}  // namespace

ModelInputs buildModelInputs(const PropellerInput& input,
                             const std::vector<Microphone>& microphones,
                             const std::vector<BladeSection>& sections,
                             const std::filesystem::path& blade_geometry_path,
                             const Metadata& metadata) {
  const int S = metadata.config.n_sections;
  const int F = metadata.config.f_per_section;
  const int base_f = F - metadata.config.station_extra_dim;
  const int batch = static_cast<int>(microphones.size());
  if (batch <= 0) {
    throw std::runtime_error("No microphones supplied.");
  }

  std::vector<double> station_base = computeBaseStationFeatures(sections, blade_geometry_path, metadata);
  std::vector<double> station_extra_raw;
  std::vector<double> global_extra_raw;
  computeExplicitGeometry(sections, blade_geometry_path, metadata, station_extra_raw, global_extra_raw);

  std::vector<double> station_one(static_cast<std::size_t>(S * F), 0.0);
  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < base_f; ++j) {
      station_one[static_cast<std::size_t>(i * F + j)] = station_base[static_cast<std::size_t>(i * base_f + j)];
    }
    for (int j = 0; j < 8; ++j) {
      const double norm = (station_extra_raw[static_cast<std::size_t>(i * 8 + j)] -
                           metadata.station_geo_mean[static_cast<std::size_t>(j)]) /
                          std::max(metadata.station_geo_std[static_cast<std::size_t>(j)], 1.0e-12);
      station_one[static_cast<std::size_t>(i * F + base_f + j)] = norm;
    }
    const double valid = station_extra_raw[static_cast<std::size_t>(i * 8 + 0)] > 0.0 ? 1.0 : 0.0;
    station_one[static_cast<std::size_t>(i * F + base_f + 8)] = valid;
  }

  std::vector<double> global_geo_norm(global_extra_raw.size(), 0.0);
  for (std::size_t i = 0; i < global_extra_raw.size(); ++i) {
    global_geo_norm[i] = (global_extra_raw[i] - metadata.global_geo_mean[i]) / std::max(metadata.global_geo_std[i], 1.0e-12);
  }

  ModelInputs out;
  out.batch = batch;
  out.station_feats.resize(static_cast<std::size_t>(batch * S * F), 0.0f);
  out.global_feats.resize(static_cast<std::size_t>(batch * metadata.config.global_dim), 0.0f);
  out.raw_global.resize(static_cast<std::size_t>(batch * metadata.config.raw_global_dim), 0.0f);
  out.bpf_hz.resize(static_cast<std::size_t>(batch), 0.0f);

  for (int b = 0; b < batch; ++b) {
    for (int k = 0; k < S * F; ++k) {
      out.station_feats[static_cast<std::size_t>(b * S * F + k)] = static_cast<float>(station_one[static_cast<std::size_t>(k)]);
    }

    const std::vector<double> g = makeRawGlobal(input, microphones[static_cast<std::size_t>(b)], metadata);
    for (int k = 0; k < metadata.config.raw_global_dim; ++k) {
      out.raw_global[static_cast<std::size_t>(b * metadata.config.raw_global_dim + k)] = static_cast<float>(g[static_cast<std::size_t>(k)]);
    }

    const double blades = std::max(g[0] * 10.0, 1.0e-3);
    const double rpm = std::max(g[4] * 8000.0, 1.0e-3);
    out.bpf_hz[static_cast<std::size_t>(b)] = static_cast<float>(blades * rpm / 60.0);

    std::vector<double> aug;
    const auto receiver = receiverFeaturesWithAngles(g);
    const auto operating = operatingFeatures(g);
    const auto directivity = directivityBasis(g);
    aug.insert(aug.end(), receiver.begin(), receiver.end());
    aug.insert(aug.end(), operating.begin(), operating.end());
    aug.insert(aug.end(), global_geo_norm.begin(), global_geo_norm.end());
    aug.insert(aug.end(), directivity.begin(), directivity.end());
    if (static_cast<int>(aug.size()) != metadata.config.global_dim) {
      throw std::runtime_error("Engineered global feature width mismatch. Built " + std::to_string(aug.size()) +
                               " but metadata expects " + std::to_string(metadata.config.global_dim));
    }
    for (int k = 0; k < metadata.config.global_dim; ++k) {
      const double z = (aug[static_cast<std::size_t>(k)] - metadata.global_aug_mean[static_cast<std::size_t>(k)]) /
                       std::max(metadata.global_aug_std[static_cast<std::size_t>(k)], 1.0e-12);
      out.global_feats[static_cast<std::size_t>(b * metadata.config.global_dim + k)] = static_cast<float>(z);
    }
  }

  return out;
}

}  // namespace lopnor
