using OcrNet.Core.Compute;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Training;

/// <summary>Training data held fully in memory (tests, MNIST evaluation, small datasets).</summary>
public sealed class InMemoryTrainingData : ITrainingData
{
    private readonly float[][] _trainInputs;
    private readonly int[] _trainLabels;
    private readonly float[][] _validationInputs;
    private readonly int[] _validationLabels;
    private List<Batch>? _validationBatches;
    private int _validationBatchSize;

    public InMemoryTrainingData(
        TensorShape inputShape, int classCount,
        float[][] trainInputs, int[] trainLabels,
        float[][] validationInputs, int[] validationLabels)
    {
        InputShape = inputShape;
        ClassCount = classCount;
        _trainInputs = trainInputs;
        _trainLabels = trainLabels;
        _validationInputs = validationInputs;
        _validationLabels = validationLabels;
    }

    public TensorShape InputShape { get; }

    public int ClassCount { get; }

    public int TrainSamplesPerEpoch => _trainInputs.Length;

    public IEnumerable<Batch> GetTrainBatches(int epoch, int batchSize, ulong seed, CancellationToken cancellationToken)
    {
        var order = Enumerable.Range(0, _trainInputs.Length).ToArray();
        new FastRandom(FastRandom.Derive(seed, epoch)).Shuffle(order.AsSpan());
        var batch = new Batch(batchSize, InputShape);
        for (int start = 0; start < order.Length; start += batchSize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(batchSize, order.Length - start);
            batch.Resize(count);
            for (int i = 0; i < count; i++)
            {
                int idx = order[start + i];
                _trainInputs[idx].CopyTo(batch.Inputs.Sample(i));
                batch.Labels[i] = _trainLabels[idx];
            }
            yield return batch;
        }
    }

    public IReadOnlyList<Batch> GetValidationBatches(int batchSize)
    {
        if (_validationBatches is not null && _validationBatchSize == batchSize)
            return _validationBatches;
        _validationBatches = ToBatches(_validationInputs, _validationLabels, InputShape, batchSize);
        _validationBatchSize = batchSize;
        return _validationBatches;
    }

    public static List<Batch> ToBatches(float[][] inputs, int[] labels, TensorShape shape, int batchSize)
    {
        var batches = new List<Batch>();
        for (int start = 0; start < inputs.Length; start += batchSize)
        {
            int count = Math.Min(batchSize, inputs.Length - start);
            var batch = new Batch(count, shape);
            for (int i = 0; i < count; i++)
            {
                inputs[start + i].CopyTo(batch.Inputs.Sample(i));
                batch.Labels[i] = labels[start + i];
            }
            batches.Add(batch);
        }
        return batches;
    }
}
