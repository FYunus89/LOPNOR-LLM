# LOPNOR_LLM C++/ONNX Inference Release

- `cpp/` is the public C++ inference program. It reads `LOPNOR_LLM.i`, `bladeGeom.txt`, `mics.txt`, the exported ONNX files, and metadata JSON.
- `examples/` contains the expected text-file layout.

## 1. The Model

The trained model is given in onnx and jason format:

```text
model/lopnor_v71_broadband.onnx
model/lopnor_v71_tonal.onnx
model/lopnor_v71_metadata.json
```

## 2. Build The C++ Program

Install ONNX Runtime C/C++ and configure CMake with its root directory:

```bash
cmake -S cpp -B cpp/build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime

cmake --build cpp/build -j
```

The CMake project uses `nlohmann_json`. If it is not already installed, CMake attempts to fetch it.

## 3. Run Prediction

```bash
cpp/build/lopnor_llm_predict \
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
11. advance ratio `J` or `AUTO`
12. microphone filename

When `J` is `AUTO`, the C++ code computes:

```text
J = V_inf / ((RPM / 60) * 2 * R_tip)
```
microphone file: first row indicates the microphon number and starting from second row contains x,y,z coordinates for each microphone

The LOPNOR model is a learned surrogate, so it is reliable mainly inside the distribution it saw during training. Source-receiver distance affects propagation loss and receiver/directivity features; beyond the largest trained distance, the model is extrapolating rather than interpolating. Neural models generally have no guarantee that this extrapolation follows the correct acoustic physics, so predictions can become unreliable even if the code still returns a number. The model should not be used for source-receiver distances larger than the largest source-receiver distance (46 times of rotor diameter) in the training data set.

## Notes

The C++ code intentionally repeats the original source code preprocessing recipe: airfoil resampling, PCA projection, explicit blade-geometry features, receiver/directivity features, operating-condition features, z-score normalization, tonal rendering, distance denormalization, one-third-octave integration, and WAV preview writing.

## Reference

```bibtex
@inproceedings{yunus2026large,
  title={Large Language Models of Propeller Noise},
  author={Yunus, Furkat},
  booktitle={32nd AIAA/CEAS Aeroacoustics Conference (2026)},
  pages={3301},
  year={2026}
}
```

# LOPNOR-LLM
Will be periodically updated, stay tuned.
