# OcrNet

OcrNet is a Windows desktop app (C#, WPF, .NET 10) for designing feed-forward neural networks,
training them on glyphs rendered from the fonts installed on your machine, and then recognizing
digits and letters that you draw with the mouse or a pen. Trained networks are saved in a compact
binary format (`.ocrnet`). A visualizer shows the network's structure, its weights, and live
activations as you draw.

The neural-network engine is written from scratch. It has no ML framework dependency and runs on
the CPU with AVX2/FMA SIMD and multithreading, so you can train a useful model on a laptop in
seconds to a few minutes.

---

## Contents

1. [Quick start](#quick-start)
2. [Using the app](#using-the-app)
3. [Training parameters](#training-parameters)
4. [Solution layout](#solution-layout)
5. [Design](#design)
6. [The `.ocrnet` model format](#the-ocrnet-model-format)
7. [Performance and accuracy](#performance-and-accuracy)
8. [Tests, benchmarks, and the smoke test](#tests-benchmarks-and-the-smoke-test)
9. [Limitations](#limitations)
10. [Future version: convolutional networks and image recognition](#future-version-convolutional-networks-and-image-recognition)

---

## Quick start

Requirements: Windows 10/11, the .NET 10 SDK, and an x64 CPU (AVX2 and FMA are used when
available, with a scalar fallback otherwise).

```powershell
dotnet build OcrNet.sln -c Release
dotnet run --project src/OcrNet.App -c Release
```

To build a bundle you can copy to another machine:

```powershell
./build.ps1                    # tests, then publishes the app to dist\OcrNet\ (needs the .NET 10 Desktop Runtime)
./build.ps1 -Zip               # also writes dist\OcrNet-<version>-win-x64.zip
./build.ps1 -SelfContained     # includes the .NET runtime (about 150 MB larger)
```

On first launch the app renders every selected font's glyphs once (about 7 seconds for
roughly 350 font faces) and caches them in `%LOCALAPPDATA%\OcrNet\glyphcache.bin`.

To train your first model:

1. **Design**: pick a preset, for example *Digits – fast* or *Balanced 47 – recommended*.
2. **Training**: click **Train from scratch** and watch the loss and accuracy charts.
3. **Recognize**: draw a character. The prediction updates every time you lift the pen.
4. **Ctrl+S**: save the model as an `.ocrnet` file. Use **Open…** to load it again later.

---

## Using the app

The window has five tabs.

### 1. Design

- **Presets**: tuned starting points that set the character set, the architecture, and the training
  settings together.

  | Preset                    | Classes | Hidden layers                      | Epochs |
  | ------------------------- | ------: | ---------------------------------- | -----: |
  | Digits – fast             |      10 | 256, 128                           |      8 |
  | Balanced 47 – recommended |      47 | 512, 256                           |     15 |
  | Alphanumeric 62           |      62 | 512, 256                           |     15 |
  | Alphanumeric 62 – large   |      62 | 1024, 512, 256 with batch norm     |     20 |

- **Output classes**:
  - *Digits* (10).
  - *Digits + uppercase* (36).
  - *Balanced* (47). This is EMNIST-style: lowercase letters that look like their capitals
    (`c i j k l m o p s u v w x y z`) are merged into one class with them.
  - *Digits + upper + lower* (62).
  - *Custom*: any set of characters, for example `0123456789+-=`.
- **Hidden layers**: set the network depth and width. Each block is
  `Dense → [BatchNorm] → activation → Dropout`, and each block has its own unit count, activation
  (ReLU, LeakyReLU, Tanh, Sigmoid, GELU), dropout rate, and batch-norm toggle. You can add,
  remove, and reorder blocks. The output layer (one unit per class, softmax) is added
  automatically.
- **Initialization**: Auto, He, Xavier, or LeCun, plus a seed. *Auto* picks He for ReLU-family
  activations and Xavier for the others.
- **Summary**: the parameter count, FLOPs per sample, and a rough per-epoch time estimate. An
  architecture diagram of the untrained network is shown next to it.
- **Guidance**: reviews the hidden layers against the selected output classes. It recommends a
  starting architecture for that number of classes and flags problems:
  - layers too narrow for the class count, which create bottlenecks;
  - designs that are too deep or too wide to be worth the extra training time;
  - saturating activations, and heavy or missing dropout;
  - a weight initialization that doesn't match the activations;
  - look-alike characters that cap the accuracy of the chosen set.

  An expandable section explains how each design parameter affects the result, and every control
  has a tooltip with the same explanation.
- **Build network**: replaces the current model with a freshly initialized one.

### 2. Dataset

- **Fonts**: lists every installed font family with a preview. Symbol fonts (Wingdings, Marlett,
  and so on) are excluded. Decorative, script, and display fonts are listed but unselected by
  default, because they make digit accuracy worse (see [accuracy](#performance-and-accuracy)).
  You can choose which style variants (regular, bold, italic, and bold italic) to render.
- **Augmentation**: rotation, shear, scale, aspect change, elastic distortion, stroke thickness
  (dilate and erode), and noise. A preview grid shows freshly augmented samples as you move the
  sliders.
- **Sampling**:
  - Samples per class per epoch.
  - The fraction of font families held out for validation.
  - How often your own drawings are mixed into training.
- **Your drawings**: characters you correct on the Recognize tab are stored in
  `%LOCALAPPDATA%\OcrNet\user-samples.bin` and mixed into training.

### 3. Training

- All hyperparameters (see the next section), plus **Train from scratch**, **Continue training**
  (which keeps the weights and the optimizer state), **Pause**, and **Stop**.
- Live tiles for epoch, loss, train and validation accuracy, learning rate, samples per second,
  and elapsed and remaining time.
- Charts of loss, accuracy (train versus validation), and the learning-rate schedule.
- When training ends, a validation confusion matrix (hover a cell for counts) and the most
  confident mistakes.
- **Evaluate on MNIST…**: if you have the MNIST IDX files (`t10k-images-idx3-ubyte` and
  `t10k-labels-idx1-ubyte`), this measures how a font-trained digit model does on real
  handwriting.

### 4. Recognize

- An ink canvas with adjustable pen width. **Esc** clears it.
- Shows the normalized 28×28 input the network sees, and the top-5 classes with their
  probabilities.
- **Line mode**: splits a drawn word into characters by vertical projection and recognizes each
  one.
- **Teach**: saves the drawing with the correct label as a user sample.
- **Font test**: renders text in a chosen font and runs it through the same pipeline.

### 5. Visualize

- **Network graph**:
  - Each layer is a column, and wide layers are sampled. You can show either the most active
    neurons or evenly spaced ones.
  - Edges show the strongest weights per neuron: teal for positive, coral for negative.
  - With a drawing on the Recognize tab, nodes light up with their activations, edge opacity
    shows the signal actually flowing (|weight × activation|), and the output column shows class
    probabilities.
- **First-layer receptive fields**: each hidden neuron's 784 input weights shown as a 28×28
  heatmap.
- **Weight distributions** for each layer, and the **training history**.

Keyboard shortcuts: **Ctrl+O** opens a model and **Ctrl+S** saves it.

---

## Training parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| Epochs | 15 (preset-dependent) | Each epoch draws fresh augmented samples. |
| Batch size | 128 | |
| Learning rate | 0.001 | |
| Optimizer | AdamW | SGD (momentum, optional Nesterov), RMSProp, Adam, AdamW. |
| Adam β₁ / β₂, ε | 0.9 / 0.999, 1e-8 | |
| SGD momentum | 0.9 | |
| RMSProp ρ | 0.9 | |
| Weight decay (L2) | 1e-4 | Coupled (added to the gradient) for SGD, RMSProp, and Adam. Decoupled for AdamW. Applies to weights only, never to biases or batch-norm parameters. |
| Dropout | per layer | An optional global override sets every dropout layer at once. |
| LR schedule | Cosine | Constant, step (every N epochs × γ), exponential, cosine. |
| Warm-up | 0.5 epochs | Linear warm-up from 0. |
| Gradient clipping | 5.0 | Clips the global L2 norm. 0 turns it off. |
| Label smoothing | 0.05 | |
| Early stopping | patience 4 | Stops when validation accuracy hasn't improved for N epochs. The best weights are restored either way. |
| Threads | `Environment.ProcessorCount` | Shared by the GEMM kernel and the data pipeline. |
| Seed | 42 | Drives initialization, shuffling, dropout, and augmentation. Runs are reproducible. |

The training options are stored in the model file, so a loaded model shows how it was trained.

---

## Solution layout

```
OcrNet.sln
Directory.Build.props            LangVersion latest, nullable, warnings as errors
src/OcrNet.Core/                 net10.0, the engine (no UI dependencies)
  Tensors/                       Tensor (batch × C×H×W, reusable buffers), TensorShape
  Compute/                       Gemm (packed AVX2/FMA SGEMM), Ops, FastRandom (xoshiro128**), Parallelism
  Layers/                        Dense, Activation, Dropout, BatchNorm1D, Parameter, LayerFactory
  Init/                          He / Xavier / LeCun
  Optimizers/                    SGD, RMSProp, Adam/AdamW, with serializable state
  Training/                      TrainingOptions, LR schedules, Trainer, Evaluator, softmax cross-entropy
  Network/                       NeuralNetwork, NetworkSpec, NetworkBuilder, ModelMetadata
  Serialization/                 ModelFile (.ocrnet), CRC-32, binary helpers
src/OcrNet.Data/                 net10.0-windows, uses WPF media for glyph rendering
  Glyphs/                        FontCatalog, GlyphRenderer, GlyphCache
  Imaging/                       GrayImage, Normalizer, Augmenter, Segmenter
  Datasets/                      CharacterSet, GlyphDataset, BatchPipeline, UserSampleStore, MnistReader
src/OcrNet.App/                  WPF app (MVVM with CommunityToolkit.Mvvm, Fluent dark theme)
  Views/, ViewModels/            one pair per tab, plus MainViewModel
  Controls/                      NetworkGraphControl, LineChart, ConfusionMatrixControl, HistogramControl
  Services/                      AppState, Presets, InkRasterizer, BitmapFactory, SnapshotRunner
tests/OcrNet.Core.Tests/         xUnit tests
benchmarks/OcrNet.Bench/         console benchmark (GEMM, MLP training, data pipeline, end-to-end)
tools/make-icon.cs               regenerates src/OcrNet.App/Assets/OcrNet.ico:
                                 dotnet run tools/make-icon.cs -- src/OcrNet.App/Assets/OcrNet.ico
build.ps1                        test + publish into dist\ (see Quick start)
```

---

## Design

### Engine

- **Batch-major math.** A dense layer computes `Y = X·Wᵀ + b` for the whole mini-batch with one
  matrix multiply. The backward pass is two more (`dX = dY·W` and `dW = dYᵀ·X`) plus a column sum
  for `db`. Every layer preallocates its buffers and reuses them, so training produces no garbage
  per batch.
- **SGEMM.** `Gemm.Multiply` supports all four transpose combinations.
  - It packs panels of A and B (K blocked at 256) and runs a register-blocked 6×16 `Vector256` FMA
    micro-kernel.
  - Work is split into 24×128 output tiles that run in `Parallel.For`.
  - It reaches about 120 GFLOPS at 1024³ and 50–65 GFLOPS on typical MLP shapes on a 4-core
    i7-6700HQ.
  - Without FMA it falls back to a scalar kernel.
  - Tests compare it with a naive reference on awkward sizes.
- **Element-wise work** uses `System.Numerics.Tensors.TensorPrimitives`.
- **Loss.** Softmax and cross-entropy are fused for numerical stability, with optional label
  smoothing.
- **Dropout** is inverted dropout, so inference needs no rescaling. A rate of 0 is a pass-through.
- **BatchNorm1D** keeps running statistics for inference and falls back to them for a batch of
  one.
- **Concurrency.** Each pass holds `NeuralNetwork.SyncRoot`, and the trainer takes it once per
  mini-batch. That lets the Recognize and Visualize tabs run predictions while a model trains.

### Data pipeline

- **Glyph rendering.** `GlyphTypeface.GetGlyphOutline` geometry is drawn into a 64×64
  `RenderTargetBitmap` on STA worker threads. Every printable ASCII character is rendered for every
  face, so changing the character set doesn't require re-rendering. A face without lowercase
  letters is skipped for lowercase classes.
- **Normalization** is shared by training and recognition, so drawings and font glyphs end up in
  the same space. It works like MNIST:
  1. Crop to the ink bounding box.
  2. Scale the longest side to 20 px with exact area resampling, keeping the aspect ratio.
  3. Place the result in a 28×28 image so that its center of mass is in the middle.
- **Augmentation** applies a random affine transform (rotation, shear, scale, aspect change), an
  elastic distortion, stroke thickening or thinning, and noise. Producer threads feed an ordered,
  deterministic batch pipeline, so augmentation overlaps with the matrix math. Each producer
  thread generates about 10,000 samples/s.
- **Validation split by font family.** The validation fonts are never seen during training, so
  validation accuracy measures how well the model generalizes to new letterforms, not how well it
  memorized the training fonts.

---

## The `.ocrnet` model format

All values are little-endian. Strings are length-prefixed UTF-8 (`BinaryWriter` 7-bit-encoded
length).

```
Container
  "OCRN"            4 bytes magic
  u16               format version (currently 1)
  u16               flags: bit 0 = optimizer state present, bit 1 = body is Brotli-compressed
  i64               body length in bytes (as stored, i.e. after compression)
  body              see below
  u32               CRC-32 (IEEE) over every preceding byte

Body
  Metadata
    string          name
    string          description
    i64             created (UTC ticks)
    string          app version
    dict            training options      (i32 count, then key/value string pairs)
    dict            preprocessing options (normalizer settings, so inference matches training)
    i32 + pairs     metrics               (string key, f64 value: ValidationAccuracy, BestEpoch, ...)
    i32 + records   epoch history         (i32 epoch, f64 train loss/acc, val loss/acc, lr, seconds)
  i32 × 3           input shape C, H, W
  i32 + strings     class labels in output order (a merged class has several characters, e.g. "Cc")
  i32               layer count, then per layer:
    u16             layer kind id
    i32 + bytes     layer config (kind-specific, e.g. Dense: inputs, outputs)
    i32             array count, then per array:
      string        name (e.g. "weight", "bias", "running_mean")
      i32 + i32[]   dims
      i32 + f32[]   values
  bool              optimizer state present
    u8              optimizer kind (1 SGD, 2 RMSProp, 3 Adam, 4 AdamW)
    ...             step count and moment arrays (lets "Continue training" resume exactly)
```

Layer kind ids are stable:

| Id | Layer |
| --: | --- |
| 1 | Dense |
| 2 | Activation |
| 3 | Dropout |
| 4 | BatchNorm1D |
| 100–106 | Reserved for the CNN version: Conv2D, MaxPool2D, AvgPool2D, Flatten, BatchNorm2D, GlobalAvgPool, Residual |

Files are written to a `.tmp` file first and then renamed, so a crash during a save never
corrupts an existing model. Loading checks the magic, the version, the length, and the CRC
before parsing anything. A file with a newer format version is rejected with a clear message.

---

## Performance and accuracy

Measured on an i7-6700HQ laptop (4 cores, 8 threads, AVX2) with 8 threads. "Validation" means
held-out font families that were never seen during training.

| Task | Network | Epochs | Time | Validation accuracy |
| --- | --- | --: | --: | --: |
| Digits (10) | 784-256-128-10 | 8 | 8 s | 98.1 % |
| Balanced (47) | 784-512-256-47 | 15 | 42 s | 92.8 % |
| Alphanumeric (62) | 784-512-256-62 | 15 | 59 s | 85.4 % |

Notes:

- **Training throughput** is about 25,000 augmented samples per second. The augmentation pipeline,
  not the matrix math, is usually the limit for these model sizes.
- **Most remaining errors are inherent look-alikes.** The 47-class set mostly confuses `O`/`0` and
  `l`/`I`/`1`. The 62-class set also confuses the case of `c/C`, `o/O`, `s/S`, and similar pairs,
  which no single-character classifier can tell apart without context. That is why *Balanced 47*
  is the recommended preset.
- **Decorative fonts matter.** With script and display fonts included, digit accuracy dropped to
  about 92 %, because many of them draw digits in ways no one handwrites. That is why they are
  unselected by default.
- **Handwriting differs from fonts.** Drawings vary more than font glyphs. Augmentation narrows
  the gap, and the **Teach** button closes it for your own handwriting: a few corrected samples
  per troublesome character help noticeably.

---

## Tests, benchmarks, and the smoke test

```powershell
dotnet test OcrNet.sln -c Release
```

The tests cover:

- GEMM against a naive reference for all transpose modes and edge sizes.
- Numerical gradient checks for every layer and for the fused loss.
- Every optimizer on a convex problem, weight-decay semantics, and the LR schedule.
- Model-file round trips (bit-identical predictions, optimizer state), detection of corrupt
  files, and rejection of files that aren't models.
- Trainer convergence, determinism, and cancellation.
- Normalizer translation and scale invariance, the segmenter, augmenter determinism, and the
  character sets.

```powershell
dotnet run --project benchmarks/OcrNet.Bench -c Release -- gemm mlp pipeline
dotnet run --project benchmarks/OcrNet.Bench -c Release -- digits balanced alnum
```

The modes are:

| Mode | What it measures |
| --- | --- |
| `gemm` | GFLOPS of the matrix multiply |
| `mlp` | Training throughput on synthetic data |
| `pipeline` | Augmentation throughput |
| `fonts` | Font discovery and rendering |
| `digits`, `balanced`, `alnum` | End-to-end training on fonts, reporting held-out-font accuracy |

**Developer smoke test.** The app can render itself offscreen, which also works without an
interactive desktop:

```powershell
OcrNet.App.exe --snapshot <folder> [--train] [--draw <text>]
```

It optionally trains the current network for 3 epochs, optionally draws `<text>` on the Recognize
canvas, then saves every tab as a PNG (plus a full-height capture of tabs that scroll), a sample
dialog as `dialog.png`, and `training-log.txt`, and exits.

---

## Limitations

- These are multilayer perceptrons, so they have no built-in translation invariance. The
  normalizer's centering does that job instead. Convolutional layers (below) would make models
  more robust and smaller.
- Line mode separates characters only where there are empty columns between them. Touching or
  overlapping characters (for example, cursive handwriting) are not split.
- Glyph rendering uses WPF, so the Data and App projects are Windows-only. The Core engine is plain
  `net10.0`.

---

## Future version: convolutional networks and image recognition

The goal of version 2 is general image classification: CIFAR-10, or any folder-per-class image
dataset. It would also use convolutions to make character recognition more robust. The engine was
built with this in mind:

- Tensors are already N×C×H×W.
- Layers declare input and output shapes.
- The trainer, optimizers, and serializer work with any layer.
- The model format reserves layer ids for the new layers.

### New layers

| Layer | Implementation plan |
| --- | --- |
| **Conv2D** (kernel, stride, padding, dilation, groups) | **im2col + the existing SGEMM.** Unfold each batch of input patches into a `(C·kH·kW) × (H'·W')` matrix and multiply it by the `filters × (C·kH·kW)` weight matrix. Backward is the transposed GEMMs plus col2im. Run im2col in parallel over the batch and channels, and reuse the buffers. Later, add direct 3×3 **Winograd F(2×2, 3×3)** kernels, which use about 2.25× fewer multiplies, for the most common case. |
| **MaxPool2D / AvgPool2D** | Direct SIMD loops. Max pooling stores the argmax for backward. |
| **BatchNorm2D** | Per-channel statistics over N×H×W. Can be folded into the preceding convolution at inference. |
| **GlobalAvgPool, Flatten** | Bridges from feature maps to dense heads. Flatten is just a reshape. |
| **Residual blocks** | A container layer that adds the input to the output of an inner stack, with an optional 1×1 projection. This enables ResNet-style networks. |
| **Depthwise-separable conv** (optional) | Smaller, mobile-style models. |

### Data

- An **image-folder loader** (`root/<class>/*.png|jpg`) with a stratified train/validation split,
  plus a **CIFAR-10 binary** reader.
- **RGB input** with per-channel mean and standard deviation, stored in the model's preprocessing
  dictionary so inference matches training.
- **Augmentation**: random crops with padding, horizontal flips, color jitter (brightness,
  contrast, saturation), cutout, and optionally mixup.
- Decoding and resizing in the existing producer-thread pipeline, with a decoded-image cache.

### Example architectures and CPU estimates (i7-6700HQ, 8 threads)

| Model | Use | Parameters | FLOPs per image (fwd) | Estimated epoch time |
| --- | --- | --: | --: | --: |
| LeNet-5 (28×28×1) | characters | ~60 k | ~0.4 M | 2–3 s per 50k samples |
| Small CNN (2×[conv32, conv32, pool] + dense 128) | characters, 47–62 classes | ~0.2 M | ~6 M | ~15 s per 50k samples |
| VGG-style 8-layer (32×32×3) | CIFAR-10 | ~1 M | ~40 M | ~1.5 min per 50k images |
| ResNet-8/20 (32×32×3) | CIFAR-10 | 0.08–0.27 M | 12–40 M | 1–2 min per 50k images |

These estimates assume about 40–60 GFLOPS sustained with a backward pass of about 2× the forward
pass. A small CNN should push held-out-font accuracy for characters past what the MLPs reach, and
a ResNet-20 should reach about 85–90 % on CIFAR-10 in roughly an hour of CPU training.

### Visualization

- **Learned filters**: first-layer kernels as image tiles (RGB for color input).
- **Feature maps**: per-layer activations for the current drawing or image, as a grid of
  channels.
- **Saliency maps and Grad-CAM**: highlight which pixels or regions drove the prediction, overlaid
  on the input.
- The network graph would treat convolutional layers as stacks of feature maps rather than columns
  of neurons.

### Performance roadmap

- Batch and channel parallelism, so small feature maps don't starve the threads.
- Fusing bias, activation, and pooling into the GEMM epilogue.
- An optional **GPU backend** through ILGPU (CUDA or OpenCL) behind a compute interface, with the
  CPU path kept as the reference.
- **ONNX export** of trained models for use in other runtimes, and possibly ONNX import for
  inference.

### File format

Version 2 of the `.ocrnet` container would add the reserved layer ids (100–106) with their configs
(kernel size, stride, padding, and channels). Version 1 readers reject version 2 files with a clear
message. The version 2 reader still loads every version 1 model.
