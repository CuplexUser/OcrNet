using OcrNet.Core.Layers;
using OcrNet.Core.Optimizers;
using OcrNet.Core.Training;

namespace OcrNet.Core.Tests;

public class OptimizerTests
{
    [Theory]
    [InlineData(OptimizerKind.Sgd, 0.05f)]
    [InlineData(OptimizerKind.RmsProp, 0.01f)]
    [InlineData(OptimizerKind.Adam, 0.05f)]
    [InlineData(OptimizerKind.AdamW, 0.05f)]
    public void Minimizes_ConvexQuadratic(OptimizerKind kind, float learningRate)
    {
        // f(w) = Σ cᵢ (wᵢ − tᵢ)², minimum at t.
        float[] target = [3f, -2f, 0.5f, 10f, -7f, 1f, 2f, 4f, -1f, 0f, 5f];
        float[] curvature = [1f, 2f, 0.5f, 1f, 3f, 1f, 1f, 2f, 1f, 1f, 0.7f];
        var p = new Parameter("w", [target.Length], applyWeightDecay: false);
        var optimizer = Optimizer.Create(new TrainingOptions { Optimizer = kind, WeightDecay = 0f, Momentum = 0.9f });

        for (int step = 0; step < 3000; step++)
        {
            for (int i = 0; i < target.Length; i++)
                p.Grad[i] = 2f * curvature[i] * (p.Value[i] - target[i]);
            optimizer.Step([p], learningRate);
        }

        for (int i = 0; i < target.Length; i++)
            Assert.True(MathF.Abs(p.Value[i] - target[i]) < 0.05f, $"{kind}: w[{i}] = {p.Value[i]}, expected {target[i]}");
    }

    [Theory]
    [InlineData(OptimizerKind.Sgd)]
    [InlineData(OptimizerKind.AdamW)]
    public void WeightDecay_ShrinksWeightsWithZeroGradient(OptimizerKind kind)
    {
        var weight = new Parameter("w", [4], applyWeightDecay: true);
        var bias = new Parameter("b", [4], applyWeightDecay: false);
        Array.Fill(weight.Value, 1f);
        Array.Fill(bias.Value, 1f);
        var optimizer = Optimizer.Create(new TrainingOptions { Optimizer = kind, WeightDecay = 0.1f, Momentum = 0f });

        for (int step = 0; step < 10; step++)
            optimizer.Step([weight, bias], 0.1f);

        Assert.All(weight.Value, v => Assert.InRange(v, 0.5f, 0.99f));
        Assert.All(bias.Value, v => Assert.Equal(1f, v));
    }

    [Fact]
    public void LearningRateSchedule_WarmsUpAndDecays()
    {
        var o = new TrainingOptions { LearningRate = 0.1f, Epochs = 10, WarmupEpochs = 1, Schedule = LrScheduleKind.Cosine, MinLearningRateFactor = 0.1f };
        Assert.True(LearningRateSchedule.At(o, 0.5) < 0.06f);
        Assert.Equal(0.1f, LearningRateSchedule.At(o, 1.0), 4);
        Assert.Equal(0.01f, LearningRateSchedule.At(o, 10.0), 4);

        o.Schedule = LrScheduleKind.Step;
        o.WarmupEpochs = 0;
        o.StepEpochs = 3;
        o.Gamma = 0.5f;
        Assert.Equal(0.1f, LearningRateSchedule.At(o, 2.9), 5);
        Assert.Equal(0.05f, LearningRateSchedule.At(o, 3.0), 5);
    }
}
