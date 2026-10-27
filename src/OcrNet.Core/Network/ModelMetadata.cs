namespace OcrNet.Core.Network;

/// <summary>One row of the training history stored with a model.</summary>
public sealed record EpochRecord(
    int Epoch, float TrainLoss, float TrainAccuracy, float ValidationLoss, float ValidationAccuracy,
    float LearningRate, double Seconds);

/// <summary>Descriptive information saved in the model file next to the weights.</summary>
public sealed class ModelMetadata
{
    public string Name { get; set; } = "Untitled";

    public string Description { get; set; } = "";

    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;

    public string AppVersion { get; set; } =
        typeof(ModelMetadata).Assembly.GetName().Version?.ToString() ?? "1.0.0";

    /// <summary>Snapshot of the <see cref="Training.TrainingOptions"/> used to train the model.</summary>
    public Dictionary<string, string> Training { get; set; } = [];

    /// <summary>Parameters of the input preprocessing (so inference matches training).</summary>
    public Dictionary<string, string> Preprocessing { get; set; } = [];

    /// <summary>Final metrics such as validation accuracy.</summary>
    public Dictionary<string, double> Metrics { get; set; } = [];

    public List<EpochRecord> History { get; set; } = [];
}
