#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "lopnor/feature_builder.hpp"
#include "lopnor/input_parser.hpp"
#include "lopnor/metadata.hpp"
#include "lopnor/onnx_runner.hpp"
#include "lopnor/output_writer.hpp"
#include "lopnor/postprocess.hpp"

namespace {

struct Args {
  std::filesystem::path menu = "LOPNOR_LLM.i";
  std::filesystem::path model_dir = "model";
  std::filesystem::path metadata = "model/lopnor_v71_metadata.json";
  std::filesystem::path out_dir = "outputs";
  double audio_seconds = 2.0;
  int audio_rate = 44100;
  bool no_audio = false;
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after " + name);
      }
      return argv[++i];
    };
    if (a == "--input") {
      args.menu = need_value(a);
    } else if (a == "--model-dir") {
      args.model_dir = need_value(a);
    } else if (a == "--metadata") {
      args.metadata = need_value(a);
    } else if (a == "--out-dir") {
      args.out_dir = need_value(a);
    } else if (a == "--audio-seconds") {
      args.audio_seconds = std::stod(need_value(a));
    } else if (a == "--audio-rate") {
      args.audio_rate = std::stoi(need_value(a));
    } else if (a == "--no-audio") {
      args.no_audio = true;
    } else if (a == "--help" || a == "-h") {
      std::cout
          << "Usage: lopnor_predict --input LOPNOR_LLM.i --model-dir model "
             "--metadata model/lopnor_v71_metadata.json --out-dir outputs\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);
    const lopnor::Metadata metadata = lopnor::readMetadata(args.metadata);
    const lopnor::PropellerInput input = lopnor::readMenuFile(args.menu);
    const auto blade_path = lopnor::resolveRelativeTo(args.menu, input.blade_geometry_file);
    const auto mic_path = lopnor::resolveRelativeTo(args.menu, input.microphones_file);
    const auto sections = lopnor::readBladeGeometry(blade_path);
    const auto microphones = lopnor::readMicrophones(mic_path);

    std::filesystem::create_directories(args.out_dir);

    const lopnor::ModelInputs model_inputs =
        lopnor::buildModelInputs(input, microphones, sections, blade_path, metadata);
    lopnor::OnnxRunner runner(args.model_dir, metadata);
    const lopnor::NeuralOutputs neural = runner.run(model_inputs, metadata);
    const auto predictions = lopnor::postprocessPredictions(neural, model_inputs, metadata);

    for (std::size_t i = 0; i < predictions.size(); ++i) {
      const std::string stem = input.case_name + "_" + lopnor::micStem(static_cast<int>(i));
      lopnor::writeSpectrumCsv(args.out_dir / (stem + "_raw_spectrum.csv"), predictions[i]);
      lopnor::writeBandCsv(args.out_dir / (stem + "_one_third_octave.csv"), predictions[i]);
      if (!args.no_audio) {
        lopnor::writePreviewWav(
            args.out_dir / (stem + "_preview.wav"),
            predictions[i],
            args.audio_seconds,
            args.audio_rate,
            static_cast<unsigned int>(1234 + i));
      }
    }

    std::cout << "Predicted " << predictions.size() << " microphone spectrum/spectra into "
              << args.out_dir << "\n";
    if (!args.no_audio) {
      std::cout << "WAV files are qualitative previews, not calibrated acoustic recordings.\n";
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "lopnor_predict failed: " << exc.what() << "\n";
    return 1;
  }
}
