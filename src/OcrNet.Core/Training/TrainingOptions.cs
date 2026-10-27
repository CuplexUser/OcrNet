using System.Globalization;
using System.Reflection;
using OcrNet.Core.Optimizers;

namespace OcrNet.Core.Training;

public enum LrScheduleKind
{
    Constant,
    Step,
    Exponential,
    Cosine,
}

/// <summary>All hyperparameters of a training run.</summary>
public sealed class TrainingOptions
{
    public int Epochs { get; set; } = 15;

    public int BatchSize { get; set; } = 128;

    public float LearningRate { get; set; } = 0.001f;

    public OptimizerKind Optimizer { get; set; } = OptimizerKind.AdamW;

    /// <summary>SGD momentum (0 = plain SGD).</summary>
    public float Momentum { get; set; } = 0.9f;

    public bool Nesterov { get; set; } = true;

    public float Beta1 { get; set; } = 0.9f;

    public float Beta2 { get; set; } = 0.999f;

    public float RmsPropRho { get; set; } = 0.9f;

    public float Epsilon { get; set; } = 1e-8f;

    /// <summary>L2 regularization strength, or decoupled weight decay for AdamW.</summary>
    public float WeightDecay { get; set; } = 1e-4f;

    /// <summary>When ≥ 0, overrides the dropout rate of every dropout layer in the network.</summary>
    public float DropoutOverride { get; set; } = -1f;

    public LrScheduleKind Schedule { get; set; } = LrScheduleKind.Cosine;

    /// <summary>Epochs between decays for <see cref="LrScheduleKind.Step"/>.</summary>
    public int StepEpochs { get; set; } = 5;

    /// <summary>Decay factor for step (per step) and exponential (per epoch) schedules.</summary>
    public float Gamma { get; set; } = 0.5f;

    /// <summary>Final learning rate as a fraction of the initial one for the cosine schedule.</summary>
    public float MinLearningRateFactor { get; set; } = 0.02f;

    public float WarmupEpochs { get; set; } = 0.5f;

    /// <summary>Global gradient-norm clipping threshold (0 = off).</summary>
    public float GradientClipNorm { get; set; } = 5f;

    public float LabelSmoothing { get; set; } = 0.05f;

    /// <summary>Stop when validation accuracy has not improved for this many epochs (0 = off).</summary>
    public int EarlyStoppingPatience { get; set; } = 4;

    /// <summary>Restore the weights of the best validation epoch when training ends.</summary>
    public bool RestoreBestWeights { get; set; } = true;

    public int Threads { get; set; } = Environment.ProcessorCount;

    public ulong Seed { get; set; } = 42;

    public TrainingOptions Clone() => (TrainingOptions)MemberwiseClone();

    private static readonly PropertyInfo[] Properties =
        typeof(TrainingOptions).GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.CanRead && p.CanWrite).ToArray();

    /// <summary>Invariant-culture key/value form, stored in model files.</summary>
    public Dictionary<string, string> ToDictionary() =>
        Properties.ToDictionary(p => p.Name, p => Convert.ToString(p.GetValue(this), CultureInfo.InvariantCulture) ?? "");

    /// <summary>Reads values written by <see cref="ToDictionary"/>; unknown or malformed keys are ignored.</summary>
    public static TrainingOptions FromDictionary(IReadOnlyDictionary<string, string> values)
    {
        var options = new TrainingOptions();
        foreach (var p in Properties)
        {
            if (!values.TryGetValue(p.Name, out var text))
                continue;
            try
            {
                object value = p.PropertyType.IsEnum
                    ? Enum.Parse(p.PropertyType, text)
                    : Convert.ChangeType(text, p.PropertyType, CultureInfo.InvariantCulture);
                p.SetValue(options, value);
            }
            catch (Exception ex) when (ex is FormatException or ArgumentException or OverflowException or InvalidCastException)
            {
                // Keep the default for values this version cannot parse.
            }
        }
        return options;
    }
}
