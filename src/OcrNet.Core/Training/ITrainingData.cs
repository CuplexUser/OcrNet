using OcrNet.Core.Tensors;

namespace OcrNet.Core.Training;

/// <summary>A mini-batch of preprocessed inputs and their class indices.</summary>
public sealed class Batch(int capacity, TensorShape shape)
{
    public Tensor Inputs { get; } = new(capacity, shape);

    public int[] Labels { get; } = new int[capacity];

    public int Count => Inputs.Batch;

    public void Resize(int count) => Inputs.Resize(count);
}

/// <summary>Source of training and validation batches.</summary>
public interface ITrainingData
{
    TensorShape InputShape { get; }

    int ClassCount { get; }

    int TrainSamplesPerEpoch { get; }

    /// <summary>
    /// Yields the shuffled training batches of one epoch. Implementations may recycle a batch
    /// once the enumerator advances, so consumers must not keep references to earlier batches.
    /// </summary>
    IEnumerable<Batch> GetTrainBatches(int epoch, int batchSize, ulong seed, CancellationToken cancellationToken);

    /// <summary>Fixed validation batches (the same every epoch).</summary>
    IReadOnlyList<Batch> GetValidationBatches(int batchSize);
}
