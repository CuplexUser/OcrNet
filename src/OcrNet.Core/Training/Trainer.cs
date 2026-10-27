using System.Diagnostics;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;
using OcrNet.Core.Compute;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Core.Optimizers;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Training;

/// <summary>Progress snapshot published while training (throttled, plus once per epoch).</summary>
public sealed record TrainingProgress(
    int Epoch,
    int Epochs,
    int BatchIndex,
    int BatchesPerEpoch,
    float Loss,
    float Accuracy,
    float LearningRate,
    double SamplesPerSecond,
    TimeSpan Elapsed,
    TimeSpan? Remaining,
    EpochRecord? CompletedEpoch = null,
    EvaluationResult? Validation = null);

public sealed class TrainingResult
{
    public required IReadOnlyList<EpochRecord> History { get; init; }

    public EvaluationResult? FinalValidation { get; init; }

    public required Optimizer Optimizer { get; init; }

    public bool Cancelled { get; init; }

    public bool StoppedEarly { get; init; }

    /// <summary>Set when training stopped because the loss became NaN/∞.</summary>
    public bool Diverged { get; init; }

    public int BestEpoch { get; init; }

    public TimeSpan Duration { get; init; }
}

/// <summary>
/// Mini-batch training loop: forward, loss, backward, gradient clipping and an optimizer step
/// per batch; validation, checkpointing and early stopping per epoch. Supports pause/resume
/// and cancellation, and can continue from a previously saved optimizer state.
/// </summary>
public sealed class Trainer
{
    private static readonly TimeSpan ReportInterval = TimeSpan.FromMilliseconds(200);

    private readonly NeuralNetwork _network;
    private readonly ITrainingData _data;
    private readonly TrainingOptions _options;
    private readonly IProgress<TrainingProgress>? _progress;
    private readonly ManualResetEventSlim _runGate = new(initialState: true);
    private readonly Optimizer _optimizer;

    public Trainer(NeuralNetwork network, ITrainingData data, TrainingOptions options,
        IProgress<TrainingProgress>? progress = null, Optimizer? resumeOptimizer = null)
    {
        if (data.InputShape.Size != network.InputShape.Size)
            throw new ArgumentException($"Data provides {data.InputShape} inputs but the network expects {network.InputShape}.");
        if (data.ClassCount != network.ClassCount)
            throw new ArgumentException($"Data has {data.ClassCount} classes but the network has {network.ClassCount}.");

        _network = network;
        _data = data;
        _options = options.Clone();
        _progress = progress;
        _optimizer = resumeOptimizer is not null && resumeOptimizer.Kind == _options.Optimizer
            ? resumeOptimizer
            : Optimizer.Create(_options);
    }

    public bool IsPaused => !_runGate.IsSet;

    public void Pause() => _runGate.Reset();

    public void Resume() => _runGate.Set();

    public Task<TrainingResult> RunAsync(CancellationToken cancellationToken) =>
        Task.Factory.StartNew(() => Run(cancellationToken), cancellationToken,
            TaskCreationOptions.LongRunning, TaskScheduler.Default);

    public TrainingResult Run(CancellationToken cancellationToken)
    {
        var o = _options;
        Parallelism.MaxThreads = o.Threads;
        var parameters = _network.Parameters.ToList();
        PrepareDropout();

        var history = _network.Metadata.History;
        int epochOffset = history.Count;
        int batchesPerEpoch = (_data.TrainSamplesPerEpoch + o.BatchSize - 1) / o.BatchSize;
        long totalBatches = (long)batchesPerEpoch * o.Epochs;
        var validation = _data.GetValidationBatches(Math.Max(o.BatchSize, 256));

        Tensor? gradient = null, probabilities = null;
        float[][]? bestWeights = null;
        float bestScore = float.NegativeInfinity;
        int bestEpoch = 0, epochsWithoutImprovement = 0;
        bool cancelled = false, stoppedEarly = false, diverged = false;
        EvaluationResult? lastEval = null;
        var total = Stopwatch.StartNew();
        long batchesDone = 0;

        try
        {
            for (int e = 0; e < o.Epochs; e++)
            {
                int epochNumber = epochOffset + e + 1;
                var epochWatch = Stopwatch.StartNew();
                var reportWatch = Stopwatch.StartNew();
                double lossSum = 0;
                long correctSum = 0, seen = 0;
                float lr = o.LearningRate;
                int batchIndex = 0;

                foreach (var batch in _data.GetTrainBatches(epochNumber, o.BatchSize, o.Seed, cancellationToken))
                {
                    _runGate.Wait(cancellationToken);
                    lr = LearningRateSchedule.At(o, e + (double)batchIndex / batchesPerEpoch);
                    float loss;
                    int correct;
                    lock (_network.SyncRoot)
                    {
                        var logits = _network.Forward(batch.Inputs, training: true);
                        var probs = Tensor.Ensure(ref probabilities, batch.Count, logits.Shape);
                        var grad = Tensor.Ensure(ref gradient, batch.Count, logits.Shape);
                        loss = SoftmaxCrossEntropy.Compute(logits, batch.Labels.AsSpan(0, batch.Count), probs, grad, o.LabelSmoothing, out correct);
                        _network.Backward(grad);
                        if (o.GradientClipNorm > 0)
                            ClipGradients(parameters, o.GradientClipNorm);
                        _optimizer.Step(parameters, lr);
                    }

                    if (!float.IsFinite(loss))
                    {
                        diverged = true;
                        break;
                    }

                    lossSum += loss * batch.Count;
                    correctSum += correct;
                    seen += batch.Count;
                    batchIndex++;
                    batchesDone++;

                    if (reportWatch.Elapsed >= ReportInterval)
                    {
                        reportWatch.Restart();
                        _progress?.Report(new TrainingProgress(
                            epochNumber, epochOffset + o.Epochs, batchIndex, batchesPerEpoch,
                            (float)(lossSum / seen), (float)correctSum / seen, lr,
                            seen / epochWatch.Elapsed.TotalSeconds, total.Elapsed,
                            Remaining(total.Elapsed, batchesDone, totalBatches)));
                    }
                }

                if (diverged)
                    break;

                lastEval = validation.Count > 0 ? Evaluator.Evaluate(_network, validation) : null;
                float trainLoss = seen > 0 ? (float)(lossSum / seen) : float.NaN;
                float trainAcc = seen > 0 ? (float)correctSum / seen : float.NaN;
                var record = new EpochRecord(epochNumber, trainLoss, trainAcc,
                    lastEval?.Loss ?? float.NaN, lastEval?.Accuracy ?? float.NaN, lr, epochWatch.Elapsed.TotalSeconds);
                history.Add(record);

                float score = lastEval?.Accuracy ?? trainAcc;
                if (score > bestScore)
                {
                    bestScore = score;
                    bestEpoch = epochNumber;
                    epochsWithoutImprovement = 0;
                    if (o.RestoreBestWeights)
                        bestWeights = _network.SnapshotWeights();
                }
                else
                {
                    epochsWithoutImprovement++;
                }

                _progress?.Report(new TrainingProgress(
                    epochNumber, epochOffset + o.Epochs, batchIndex, batchesPerEpoch, trainLoss, trainAcc, lr,
                    seen / epochWatch.Elapsed.TotalSeconds, total.Elapsed,
                    Remaining(total.Elapsed, batchesDone, totalBatches), record, lastEval));

                if (o.EarlyStoppingPatience > 0 && epochsWithoutImprovement >= o.EarlyStoppingPatience)
                {
                    stoppedEarly = true;
                    break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            cancelled = true;
        }

        bool restored = false;
        if (bestWeights is not null && history.Count > 0 && history[^1].Epoch != bestEpoch)
        {
            _network.RestoreWeights(bestWeights);
            restored = true;
        }
        if ((restored || lastEval is null) && validation.Count > 0)
            lastEval = Evaluator.Evaluate(_network, validation);

        _network.Metadata.Training = o.ToDictionary();
        if (lastEval is not null)
        {
            _network.Metadata.Metrics["ValidationAccuracy"] = lastEval.Accuracy;
            _network.Metadata.Metrics["ValidationLoss"] = lastEval.Loss;
        }
        _network.Metadata.Metrics["BestEpoch"] = bestEpoch;
        _network.Metadata.Metrics["TrainingSeconds"] =
            _network.Metadata.Metrics.GetValueOrDefault("TrainingSeconds") + total.Elapsed.TotalSeconds;

        return new TrainingResult
        {
            History = history,
            FinalValidation = lastEval,
            Optimizer = _optimizer,
            Cancelled = cancelled,
            StoppedEarly = stoppedEarly,
            Diverged = diverged,
            BestEpoch = bestEpoch,
            Duration = total.Elapsed,
        };
    }

    private void PrepareDropout()
    {
        int index = 0;
        foreach (var dropout in _network.Layers.OfType<DropoutLayer>())
        {
            if (_options.DropoutOverride >= 0)
                dropout.Rate = _options.DropoutOverride;
            dropout.Seed(FastRandom.Derive(_options.Seed, 7, index++));
        }
    }

    private static void ClipGradients(List<Parameter> parameters, float maxNorm)
    {
        double sumSquares = 0;
        foreach (var p in parameters)
            sumSquares += TensorPrimitives.SumOfSquares<float>(p.Grad);
        double norm = Math.Sqrt(sumSquares);
        if (norm <= maxNorm || !double.IsFinite(norm))
            return;
        float scale = (float)(maxNorm / norm);
        foreach (var p in parameters)
            TensorPrimitives.Multiply(p.Grad, scale, p.Grad);
    }

    private static TimeSpan? Remaining(TimeSpan elapsed, long done, long total)
    {
        if (done == 0)
            return null;
        return TimeSpan.FromSeconds(elapsed.TotalSeconds / done * Math.Max(0, total - done));
    }
}
