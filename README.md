# LOPNOR_LLM C++/ONNX Inference Release

This folder separates deployment from training.

- `cpp/` is the public C++ inference program. It reads `LOPNOR_LLM.i`, `bladeGeom.txt`, `mics.txt`, the exported ONNX files, and metadata JSON.
- `examples/` contains the expected text-file layout.

## 1. The Model

Model output files:

```text
PropNoise/lopnor_release/model/lopnor_v71_broadband.onnx
PropNoise/lopnor_release/model/lopnor_v71_tonal.onnx
PropNoise/lopnor_release/model/lopnor_v71_metadata.json
```
which are located in model folder.

## 2. Build The C++ Program

Install ONNX Runtime C/C++ and configure CMake with its root directory:

```bash
cmake -S PropNoise/lopnor_release/cpp -B PropNoise/lopnor_release/cpp/build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime

cmake --build PropNoise/lopnor_release/cpp/build -j
```

The CMake project uses `nlohmann_json`. If it is not already installed, CMake attempts to fetch it.

## 3. Run Prediction

```bash
PropNoise/lopnor_release/cpp/build/lopnor_predict \
  --input PropNoise/lopnor_release/examples/LOPNOR_LLM.i \
  --model-dir PropNoise/lopnor_release/model \
  --metadata PropNoise/lopnor_release/model/lopnor_v71_metadata.json \
  --out-dir PropNoise/lopnor_release/outputs
```

Outputs per microphone:

```text
*_raw_spectrum.csv
*_one_third_octave.csv
*_preview.wav
```

The `.wav` files are qualitative listening previews. They are normalized audio renderings of the predicted spectrum, not calibrated acoustic recordings.

## Required Menu Fields

`LOPNOR_LLM.i` must provide, in order:

1. case name
2. blade count
3. tip radius `[m]`
4. hub radius `[m]`
5. blade pitch angle `[deg]`
6. blade geometry filename
7. RPM
8. shaft pitch angle `[deg]`
9. shaft yaw angle `[deg]`
10. free-stream velocity `[m/s]`
11. altitude `[m]`
12. advance ratio `J` or `AUTO`
13. microphone filename

When `J` is `AUTO`, the C++ code computes:

```text
J = V_inf / ((RPM / 60) * 2 * R_tip)
```

## Notes

The C++ code intentionally repeats the original source code preprocessing recipe: airfoil resampling, PCA projection, explicit blade-geometry features, receiver/directivity features, operating-condition features, z-score normalization, tonal rendering, distance denormalization, one-third-octave integration, and WAV preview writing.

## Reference

@inproceedings{yunus2026large,
  title={Large Language Models of Propeller Noise},
  author={Yunus, Furkat},
  booktitle={32nd AIAA/CEAS Aeroacoustics Conference (2026)},
  pages={3301},
  year={2026}
}
