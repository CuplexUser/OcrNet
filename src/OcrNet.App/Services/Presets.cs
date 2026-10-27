using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Core.Optimizers;
using OcrNet.Core.Training;
using OcrNet.Data.Datasets;

namespace OcrNet.App.Services;

/// <summary>A tuned starting point: character set, architecture and training settings.</summary>
public sealed record Preset(
    string Name, string Description, CharacterSet CharacterSet, HiddenLayerSpec[] HiddenLayers,
    int Epochs, int SamplesPerClass, float LearningRate)
{
    public override string ToString() => Name;

    public TrainingOptions ApplyTo(TrainingOptions options)
    {
        var o = options.Clone();
        o.Epochs = Epochs;
        o.LearningRate = LearningRate;
        o.Optimizer = OptimizerKind.AdamW;
        o.Schedule = LrScheduleKind.Cosine;
        o.BatchSize = 128;
        return o;
    }

    private static HiddenLayerSpec Layer(int units, float dropout = 0.2f, bool batchNorm = false) =>
        new() { Units = units, Activation = ActivationKind.ReLU, Dropout = dropout, BatchNorm = batchNorm };

    public static IReadOnlyList<Preset> All { get; } =
    [
        new("Digits – fast", "0–9 with a small two-layer MLP. Trains in well under a minute.",
            CharacterSet.DigitsOnly, [Layer(256), Layer(128)], 8, 2000, 1e-3f),
        new("Balanced 47 – recommended", "Digits and letters with look-alike lowercase merged into capitals (EMNIST style).",
            CharacterSet.Balanced, [Layer(512), Layer(256)], 15, 1200, 1e-3f),
        new("Alphanumeric 62", "Every digit, capital and lowercase letter as its own class. Case look-alikes (c/C, o/O) stay ambiguous.",
            CharacterSet.Alphanumeric, [Layer(512), Layer(256)], 15, 1200, 1e-3f),
        new("Alphanumeric 62 – large", "Deeper network with batch normalization; a few minutes of training.",
            CharacterSet.Alphanumeric, [Layer(1024, 0.25f, true), Layer(512, 0.25f, true), Layer(256, 0.2f, true)], 20, 1500, 1e-3f),
    ];
}
