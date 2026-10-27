using System.IO;
using OcrNet.Core.Compute;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Core.Optimizers;
using OcrNet.Core.Serialization;
using OcrNet.Core.Tensors;
using OcrNet.Core.Training;

namespace OcrNet.Core.Tests;

public class SerializationTests
{
    private static NeuralNetwork CreateNetwork()
    {
        var spec = new NetworkSpec
        {
            InputShape = new TensorShape(1, 4, 4),
            Labels = ["0", "1", "Cc"],
            HiddenLayers =
            [
                new HiddenLayerSpec { Units = 8, Activation = ActivationKind.ReLU, Dropout = 0.25f, BatchNorm = true },
                new HiddenLayerSpec { Units = 5, Activation = ActivationKind.Tanh, Dropout = 0f },
            ],
        };
        var network = NetworkBuilder.Build(spec);
        network.Metadata.Name = "test";
        network.Metadata.Preprocessing["Size"] = "28";
        network.Metadata.Metrics["ValidationAccuracy"] = 0.9;
        network.Metadata.History.Add(new EpochRecord(1, 1.2f, 0.5f, 1.1f, 0.6f, 0.001f, 3.5));
        network.Metadata.Training = new TrainingOptions { Optimizer = OptimizerKind.Adam, LearningRate = 0.01f }.ToDictionary();
        return network;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void RoundTrip_PreservesPredictionsAndMetadata(bool compress)
    {
        var network = CreateNetwork();
        var input = new float[16];
        var rng = new FastRandom(3);
        for (int i = 0; i < input.Length; i++)
            input[i] = rng.NextFloat();
        var expected = network.Predict(input);

        using var stream = new MemoryStream();
        ModelFile.Save(stream, network, compress: compress);
        stream.Position = 0;
        var loaded = ModelFile.Load(stream).Network;

        Assert.Equal(expected, loaded.Predict(input));
        Assert.Equal(network.Labels, loaded.Labels);
        Assert.Equal(network.InputShape, loaded.InputShape);
        Assert.Equal(network.Layers.Select(l => l.DisplayName), loaded.Layers.Select(l => l.DisplayName));
        Assert.Equal("test", loaded.Metadata.Name);
        Assert.Equal("28", loaded.Metadata.Preprocessing["Size"]);
        Assert.Equal(0.9, loaded.Metadata.Metrics["ValidationAccuracy"]);
        Assert.Equal(network.Metadata.History, loaded.Metadata.History);
        Assert.Equal(0.01f, TrainingOptions.FromDictionary(loaded.Metadata.Training).LearningRate);
    }

    [Fact]
    public void RoundTrip_RestoresOptimizerState()
    {
        var network = CreateNetwork();
        var optimizer = new AdamOptimizer(0.9f, 0.999f, 1e-8f, 0f, decoupled: false);
        var parameters = network.Parameters.ToList();
        foreach (var p in parameters)
            Array.Fill(p.Grad, 0.1f);
        optimizer.Step(parameters, 0.01f);

        using var stream = new MemoryStream();
        ModelFile.Save(stream, network, optimizer);
        stream.Position = 0;
        var loaded = ModelFile.Load(stream);

        Assert.NotNull(loaded.Optimizer);
        Assert.Equal(OptimizerKind.Adam, loaded.Optimizer!.Kind);
        Assert.Equal(1, loaded.Optimizer.StepCount);
    }

    [Fact]
    public void Load_DetectsCorruption()
    {
        using var stream = new MemoryStream();
        ModelFile.Save(stream, CreateNetwork(), compress: false);
        var bytes = stream.ToArray();
        bytes[bytes.Length / 2] ^= 0x5A;

        var ex = Assert.Throws<InvalidDataException>(() => ModelFile.Load(new MemoryStream(bytes)));
        Assert.Contains("CRC", ex.Message);
    }

    [Fact]
    public void Load_RejectsForeignFile()
    {
        Assert.Throws<InvalidDataException>(() => ModelFile.Load(new MemoryStream("hello world, not a model"u8.ToArray())));
    }

    [Fact]
    public void TrainingOptions_DictionaryRoundTrip()
    {
        var o = new TrainingOptions { Epochs = 3, LearningRate = 0.123f, Optimizer = OptimizerKind.RmsProp, Nesterov = false, Seed = 77 };
        var copy = TrainingOptions.FromDictionary(o.ToDictionary());
        Assert.Equal(3, copy.Epochs);
        Assert.Equal(0.123f, copy.LearningRate);
        Assert.Equal(OptimizerKind.RmsProp, copy.Optimizer);
        Assert.False(copy.Nesterov);
        Assert.Equal(77UL, copy.Seed);
    }
}
