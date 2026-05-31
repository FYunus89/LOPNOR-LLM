#include "lopnor/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lopnor {
namespace {

double dbToPower(double db) {
  return std::pow(10.0, 0.1 * db);
}

double powerToDb(double p, double eps) {
  return 10.0 * std::log10(std::max(p, eps));
}

double receiverDistanceFromRaw(const ModelInputs& inputs, const Metadata& metadata, int batch_index) {
  const int G = metadata.config.raw_global_dim;
  const int offset = batch_index * G;
  const double x = inputs.raw_global[static_cast<std::size_t>(offset + G - 3)];
  const double y = inputs.raw_global[static_cast<std::size_t>(offset + G - 2)];
  const double z = inputs.raw_global[static_cast<std::size_t>(offset + G - 1)];
  return std::max(std::sqrt(x * x + y * y + z * z), 1.0e-6);
}

double harmonicCenterPosition(const std::vector<double>& freq, double center_hz) {
  const double min_hz = std::max(freq.front(), 1.0e-6);
  const double max_hz = std::max(freq.back(), min_hz);
  const double clamped = std::max(min_hz, std::min(max_hz, center_hz));
  const double log_center = std::log(std::max(clamped, 1.0e-6));
  std::vector<double> log_freq(freq.size());
  for (std::size_t i = 0; i < freq.size(); ++i) {
    log_freq[i] = std::log(std::max(freq[i], 1.0e-6));
  }
  auto it = std::lower_bound(log_freq.begin(), log_freq.end(), log_center);
  int hi = static_cast<int>(std::distance(log_freq.begin(), it));
  hi = std::max(1, std::min(hi, static_cast<int>(freq.size()) - 1));
  const int lo = hi - 1;
  const double frac = (log_center - log_freq[static_cast<std::size_t>(lo)]) /
                      std::max(log_freq[static_cast<std::size_t>(hi)] - log_freq[static_cast<std::size_t>(lo)], 1.0e-8);
  return std::max(0.0, std::min(static_cast<double>(freq.size() - 1), static_cast<double>(lo) + frac));
}

std::vector<double> renderTonalPsd(const std::vector<double>& line_db_actual,
                                   const std::vector<double>& line_width_bins,
                                   double bpf_hz,
                                   const Metadata& metadata) {
  const int H = metadata.config.harmonic_count;
  const int N = metadata.config.raw_grid_bins;
  std::vector<double> tonal_power(static_cast<std::size_t>(N), 0.0);
  const double mix = metadata.tonal_pseudovoigt_mix;
  for (int h = 0; h < H; ++h) {
    const double center_hz = bpf_hz * static_cast<double>(h + 1);
    if (center_hz < metadata.raw_freq_hz.front() || center_hz > metadata.raw_freq_hz.back()) {
      continue;
    }
    const double center_pos = harmonicCenterPosition(metadata.raw_freq_hz, center_hz);
    const double sigma = std::max(metadata.tonal_width_floor_bins,
                                  std::min(metadata.tonal_width_ceil_bins, line_width_bins[static_cast<std::size_t>(h)]));
    std::vector<double> kernel(static_cast<std::size_t>(N), 0.0);
    double ksum = 0.0;
    for (int i = 0; i < N; ++i) {
      const double dist = (static_cast<double>(i) - center_pos) / std::max(sigma, 1.0e-6);
      if (std::abs(dist) > 6.0) {
        continue;
      }
      const double gaussian = std::exp(-0.5 * dist * dist);
      const double lorentz = 1.0 / (1.0 + dist * dist);
      const double value = (1.0 - mix) * gaussian + mix * lorentz;
      kernel[static_cast<std::size_t>(i)] = value;
      ksum += value;
    }
    if (ksum <= 1.0e-12) {
      continue;
    }
    const double line_power = dbToPower(line_db_actual[static_cast<std::size_t>(h)]);
    for (int i = 0; i < N; ++i) {
      tonal_power[static_cast<std::size_t>(i)] += line_power * kernel[static_cast<std::size_t>(i)] / ksum;
    }
  }

  std::vector<double> tonal_psd(static_cast<std::size_t>(N), 0.0);
  for (int i = 0; i < N; ++i) {
    const double width = std::max(metadata.raw_high_hz[static_cast<std::size_t>(i)] -
                                      metadata.raw_low_hz[static_cast<std::size_t>(i)],
                                  1.0e-12);
    tonal_psd[static_cast<std::size_t>(i)] = tonal_power[static_cast<std::size_t>(i)] / width;
  }
  return tonal_psd;
}

std::vector<double> integrateBands(const std::vector<double>& raw_db, const Metadata& metadata) {
  std::vector<double> bands(metadata.target_freqs_hz.size(), 0.0);
  for (std::size_t j = 0; j < metadata.target_freqs_hz.size(); ++j) {
    const double center = metadata.target_freqs_hz[j];
    const double band_lo = center / std::pow(2.0, 1.0 / 6.0);
    const double band_hi = center * std::pow(2.0, 1.0 / 6.0);
    double power = 0.0;
    for (std::size_t i = 0; i < raw_db.size(); ++i) {
      const double lo = std::max(metadata.raw_low_hz[i], band_lo);
      const double hi = std::min(metadata.raw_high_hz[i], band_hi);
      const double overlap = std::max(hi - lo, 0.0);
      if (overlap > 0.0) {
        power += dbToPower(raw_db[i]) * overlap;
      }
    }
    bands[j] = powerToDb(power, metadata.power_eps);
  }
  return bands;
}

}  // namespace

std::vector<Prediction> postprocessPredictions(const NeuralOutputs& neural,
                                               const ModelInputs& inputs,
                                               const Metadata& metadata) {
  const int B = inputs.batch;
  const int N = metadata.config.raw_grid_bins;
  const int H = metadata.config.harmonic_count;
  if (static_cast<int>(neural.broadband_scaled.size()) != B * N ||
      static_cast<int>(neural.line_db_ref.size()) != B * H ||
      static_cast<int>(neural.line_width_bins.size()) != B * H) {
    throw std::runtime_error("Unexpected ONNX output tensor size.");
  }

  std::vector<Prediction> out(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b) {
    const double r = receiverDistanceFromRaw(inputs, metadata, b);
    const double correction = 20.0 * std::log10(std::max(r / metadata.reference_distance_m, 1.0e-6));
    Prediction pred;
    pred.raw_freq_hz = metadata.raw_freq_hz;
    pred.one_third_freq_hz = metadata.target_freqs_hz;
    pred.total_db.resize(static_cast<std::size_t>(N));
    pred.broadband_db.resize(static_cast<std::size_t>(N));
    pred.tonal_db.resize(static_cast<std::size_t>(N));
    pred.line_db.resize(static_cast<std::size_t>(H));
    std::vector<double> line_width(static_cast<std::size_t>(H));

    for (int i = 0; i < N; ++i) {
      const std::size_t idx = static_cast<std::size_t>(b * N + i);
      const double broad_ref = metadata.total_raw_mean_db[static_cast<std::size_t>(i)] +
                               metadata.total_raw_std_db[static_cast<std::size_t>(i)] *
                                   static_cast<double>(neural.broadband_scaled[idx]);
      pred.broadband_db[static_cast<std::size_t>(i)] = broad_ref - correction;
    }
    for (int h = 0; h < H; ++h) {
      const std::size_t idx = static_cast<std::size_t>(b * H + h);
      pred.line_db[static_cast<std::size_t>(h)] = static_cast<double>(neural.line_db_ref[idx]) - correction;
      line_width[static_cast<std::size_t>(h)] = static_cast<double>(neural.line_width_bins[idx]);
    }

    const double bpf = static_cast<double>(inputs.bpf_hz[static_cast<std::size_t>(b)]);
    const std::vector<double> tonal_psd = renderTonalPsd(pred.line_db, line_width, bpf, metadata);
    for (int i = 0; i < N; ++i) {
      pred.tonal_db[static_cast<std::size_t>(i)] = powerToDb(tonal_psd[static_cast<std::size_t>(i)], metadata.power_eps);
      const double total_power = dbToPower(pred.broadband_db[static_cast<std::size_t>(i)]) + tonal_psd[static_cast<std::size_t>(i)];
      pred.total_db[static_cast<std::size_t>(i)] = powerToDb(total_power, metadata.power_eps);
    }
    pred.one_third_total_db = integrateBands(pred.total_db, metadata);
    out[static_cast<std::size_t>(b)] = std::move(pred);
  }
  return out;
}

}  // namespace lopnor
