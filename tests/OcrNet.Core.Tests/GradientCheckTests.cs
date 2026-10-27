using OcrNet.Core.Compute;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Core.Tensors;
using OcrNet.Core.Training;

namespace OcrNet.Core.Tests;

/// <summary>Compares back-propagated gradients with central finite differences.</summary>
public class GradientCheckTests
{
    private const int Batch = 4;
    private const int Inputs = 6;
    private const int Classes = 3;

    [Theory]
    [InlineData(ActivationKind.ReLU, false)]
    [InlineData(ActivationKind.LeakyReLU, false)]
    [InlineData(ActivationKind.Tanh, false)]
    [InlineData(ActivationKind.Sigmoid, false)]
    [InlineData(ActivationKind.Gelu, false)]
    [InlineData(ActivationKind.Tanh, true)]
    [InlineData(ActivationKind.ReLU, true)]
    public void AnalyticGradients_MatchNumericalGradients(ActivationKind activation, bool batchNorm)
    {
        var spec = new NetworkSpec
        {
            InputShape = TensorShape.Vector(Inputs),
            Labels = ["a", "b", "c"],
            HiddenLayers =
            [
                new HiddenLayerSpec { Units = 5, Activation = activation, BatchNorm = batchNorm, Dropout = 0 },
                new HiddenLayerSpec { Units = 4, Activation = activation, BatchNorm = batchNorm, Dropout = 0 },
            ],
            Seed = 7,
        };
        var network = NetworkBuilder.Build(spec);
        var rng = new FastRandom(99);
        var input = new Tensor(Batch, spec.InputShape);
        for (int i = 0; i < input.Length; i++)
            input.Data[i] = rng.NextFloat(-1.5f, 1.5f);
        int[] labels = [0, 2, 1, 2];

        // Analytic gradients.
        var probs = new Tensor(Batch, TensorShape.Vector(Classes));
        var grad = new Tensor(Batch, TensorShape.Vector(Classes));
        var logits = network.Forward(input, training: true);
        SoftmaxCrossEntropy.Compute(logits, labels, probs, grad, 0.1f, out _);
        network.Backward(grad);

        foreach (var p in network.Parameters)
        {
            var analytic = (float[])p.Grad.Clone();
            for (int i = 0; i < p.Length; i++)
            {
                float original = p.Value[i];
                const float h = 1e-3f;
                p.Value[i] = original + h;
                double lossPlus = Loss(network, input, labels, probs);
                p.Value[i] = original - h;
                double lossMinus = Loss(network, input, labels, probs);
                p.Value[i] = original;

                double numeric = (lossPlus - lossMinus) / (2 * h);
                double tolerance = 2e-3 + 0.05 * Math.Abs(numeric);
                Assert.True(Math.Abs(numeric - analytic[i]) <= tolerance,
                    $"{p.Name}[{i}] ({activation}, bn={batchNorm}): analytic {analytic[i]:G5} vs numeric {numeric:G5}");
            }
        }
    }

    private static double Loss(NeuralNetwork network, Tensor input, int[] labels, Tensor probs)
    {
        var logits = network.Forward(input, training: true);
        return SoftmaxCrossEntropy.Compute(logits, labels, probs, null, 0.1f, out _);
    }

    [Fact]
    public void Dropout_PreservesExpectedActivation()
    {
        var layer = new DropoutLayer(TensorShape.Vector(10_000), 0.3f);
        layer.Seed(1);
        var input = new Tensor(1, TensorShape.Vector(10_000));
        input.Span.Fill(1f);

        var training = layer.Forward(input, training: true);
        float mean = training.Span.ToArray().Average();
        Assert.InRange(mean, 0.95f, 1.05f);
        int zeros = training.Span.ToArray().Count(v => v == 0f);
        Assert.InRange(zeros, 2700, 3300);

        var inference = layer.Forward(input, training: false);
        Assert.Same(input, inference);
    }
}
