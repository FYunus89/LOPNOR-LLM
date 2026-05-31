#include "lopnor/output_writer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace lopnor {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

double dbToPower(double db) {
  return std::pow(10.0, 0.1 * db);
}

void writeLittleEndian16(std::ofstream& out, std::int16_t v) {
  out.put(static_cast<char>(v & 0xff));
  out.put(static_cast<char>((v >> 8) & 0xff));
}

void writeLittleEndian32(std::ofstream& out, std::int32_t v) {
  out.put(static_cast<char>(v & 0xff));
  out.put(static_cast<char>((v >> 8) & 0xff));
  out.put(static_cast<char>((v >> 16) & 0xff));
  out.put(static_cast<char>((v >> 24) & 0xff));
}

}  // namespace

std::string micStem(int index_zero_based) {
  return "mic" + std::to_string(index_zero_based + 1);
}

void writeSpectrumCsv(const std::filesystem::path& path, const Prediction& prediction) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot write spectrum CSV: " + path.string());
  }
  out << "frequency_hz,total_db,broadband_db,tonal_db\n";
  out << std::setprecision(10);
  for (std::size_t i = 0; i < prediction.raw_freq_hz.size(); ++i) {
    out << prediction.raw_freq_hz[i] << ',' << prediction.total_db[i] << ',' << prediction.broadband_db[i] << ','
        << prediction.tonal_db[i] << '\n';
  }
}

void writeBandCsv(const std::filesystem::path& path, const Prediction& prediction) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot write one-third-octave CSV: " + path.string());
  }
  out << "center_frequency_hz,total_db\n";
  out << std::setprecision(10);
  for (std::size_t i = 0; i < prediction.one_third_freq_hz.size(); ++i) {
    out << prediction.one_third_freq_hz[i] << ',' << prediction.one_third_total_db[i] << '\n';
  }
}

void writePreviewWav(const std::filesystem::path& path,
                     const Prediction& prediction,
                     double seconds,
                     int sample_rate,
                     unsigned int seed) {
  const int n_samples = std::max(1, static_cast<int>(std::round(seconds * static_cast<double>(sample_rate))));
  std::vector<double> samples(static_cast<std::size_t>(n_samples), 0.0);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * kPi);

  for (std::size_t k = 0; k < prediction.raw_freq_hz.size(); ++k) {
    const double f = prediction.raw_freq_hz[k];
    if (f <= 20.0 || f >= 0.48 * static_cast<double>(sample_rate)) {
      continue;
    }
    const double amp = std::sqrt(std::max(dbToPower(prediction.total_db[k]), 0.0));
    const double phase = phase_dist(rng);
    const double omega = 2.0 * kPi * f / static_cast<double>(sample_rate);
    for (int n = 0; n < n_samples; ++n) {
      samples[static_cast<std::size_t>(n)] += amp * std::sin(omega * static_cast<double>(n) + phase);
    }
  }

  double max_abs = 0.0;
  for (double v : samples) {
    max_abs = std::max(max_abs, std::abs(v));
  }
  const double scale = max_abs > 1.0e-12 ? 0.85 / max_abs : 1.0;

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Cannot write WAV file: " + path.string());
  }
  const int bytes_per_sample = 2;
  const int channels = 1;
  const int data_bytes = n_samples * channels * bytes_per_sample;
  out.write("RIFF", 4);
  writeLittleEndian32(out, 36 + data_bytes);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  writeLittleEndian32(out, 16);
  writeLittleEndian16(out, 1);
  writeLittleEndian16(out, static_cast<std::int16_t>(channels));
  writeLittleEndian32(out, sample_rate);
  writeLittleEndian32(out, sample_rate * channels * bytes_per_sample);
  writeLittleEndian16(out, static_cast<std::int16_t>(channels * bytes_per_sample));
  writeLittleEndian16(out, 16);
  out.write("data", 4);
  writeLittleEndian32(out, data_bytes);
  for (double v : samples) {
    const double clipped = std::max(-1.0, std::min(1.0, v * scale));
    writeLittleEndian16(out, static_cast<std::int16_t>(std::round(clipped * 32767.0)));
  }
}

}  // namespace lopnor
