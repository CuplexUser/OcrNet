using OcrNet.Core.Compute;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Core.Optimizers;
using OcrNet.Core.Tensors;
using OcrNet.Core.Training;

namespace OcrNet.Core.Tests;

public class TrainerTests
{
    /// <summary>Four Gaussian blobs in 8-D: a small problem any working trainer solves quickly.</summary>
    private static InMemoryTrainingData CreateBlobs(int perClass, ulong seed)
    {
        var rng = new FastRandom(seed);
        const int dims = 8, classes = 4;
        var centers = new float[classes][];
        var centerRng = new FastRandom(1);
        for (int c = 0; c < classes; c++)
        {
            centers[c] = new float[dims];
            for (int d = 0; d < dims; d++)
                centers[c][d] = centerRng.NextFloat(-3f, 3f);
        }

        (float[][] x, int[] y) Make(int count)
        {
            var x = new float[count * classes][];
            var y = new int[count * classes];
            for (int i = 0; i < x.Length; i++)
            {
                int c = i % classes;
                x[i] = new float[dims];
                for (int d = 0; d < dims; d++)
                    x[i][d] = centers[c][d] + rng.NextGaussian() * 0.8f;
                y[i] = c;
            }
            return (x, y);
        }

        var train = Make(perClass);
        var val = Make(perClass / 4);
        return new InMemoryTrainingData(TensorShape.Vector(dims), classes, train.x, train.y, val.x, val.y);
    }

    [Theory]
    [InlineData(OptimizerKind.Sgd, 0.05f)]
    [InlineData(OptimizerKind.Adam, 0.005f)]
    [InlineData(OptimizerKind.AdamW, 0.005f)]
    [InlineData(OptimizerKind.RmsProp, 0.002f)]
    public void Train_LearnsSeparableBlobs(OptimizerKind optimizer, float learningRate)
    {
        var data = CreateBlobs(400, 11);
        var network = NetworkBuilder.Build(new NetworkSpec
        {
            InputShape = data.InputShape,
            Labels = ["a", "b", "c", "d"],
            HiddenLayers = [new HiddenLayerSpec { Units = 32, Dropout = 0.1f, BatchNorm = true }],
        });
        var options = new TrainingOptions
        {
            Epochs = 8, BatchSize = 32, LearningRate = learningRate, Optimizer = optimizer, EarlyStoppingPatience = 0,
        };

        var result = new Trainer(network, data, options).Run(CancellationToken.None);

        Assert.False(result.Diverged);
        Assert.Equal(8, result.History.Count);
        Assert.True(result.FinalValidation!.Accuracy > 0.95f, $"Validation accuracy {result.FinalValidation.Accuracy}");
        Assert.True(result.History[^1].TrainLoss < result.History[0].TrainLoss);
    }

    [Fact]
    public void Train_IsDeterministicForSameSeed()
    {
        float[] Run()
        {
            var data = CreateBlobs(100, 5);
            var network = NetworkBuilder.Build(new NetworkSpec
            {
                InputShape = data.InputShape,
                Labels = ["a", "b", "c", "d"],
                HiddenLayers = [new HiddenLayerSpec { Units = 16, Dropout = 0.3f }],
            });
            new Trainer(network, data, new TrainingOptions { Epochs = 2, BatchSize = 16 }).Run(CancellationToken.None);
            return network.Predict(new float[8]);
        }

        Assert.Equal(Run(), Run());
    }

    [Fact]
    public void Train_StopsWhenCancelled()
    {
        var data = CreateBlobs(400, 3);
        var network = NetworkBuilder.Build(new NetworkSpec
        {
            InputShape = data.InputShape,
            Labels = ["a", "b", "c", "d"],
            HiddenLayers = [new HiddenLayerSpec { Units = 16 }],
        });
        using var cts = new CancellationTokenSource();
        var progress = new SyncProgress(p => { if (p.CompletedEpoch?.Epoch == 1) cts.Cancel(); });

        var result = new Trainer(network, data, new TrainingOptions { Epochs = 50, BatchSize = 16 }, progress).Run(cts.Token);

        Assert.True(result.Cancelled);
        Assert.True(result.History.Count < 50);
    }

    private sealed class SyncProgress(Action<TrainingProgress> handler) : IProgress<TrainingProgress>
    {
        public void Report(TrainingProgress value) => handler(value);
    }
}
