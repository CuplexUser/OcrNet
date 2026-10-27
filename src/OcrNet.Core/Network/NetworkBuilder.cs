using OcrNet.Core.Compute;
using OcrNet.Core.Init;
using OcrNet.Core.Layers;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Network;

public static class NetworkBuilder
{
    /// <summary>
    /// Builds an MLP from <paramref name="spec"/>. Every hidden block gets a dropout layer (rate 0 is a
    /// pass-through) so the training dropout setting can be changed without rebuilding the network.
    /// </summary>
    public static NeuralNetwork Build(NetworkSpec spec, ModelMetadata? metadata = null)
    {
        if (spec.Labels.Count < 2)
            throw new ArgumentException("A classifier needs at least two classes.", nameof(spec));

        var rng = new FastRandom(spec.Seed);
        var layers = new List<Layer>();
        var shape = spec.InputShape;

        foreach (var hidden in spec.HiddenLayers)
        {
            var dense = new DenseLayer(shape, hidden.Units);
            dense.Initialize(ResolveInit(spec.WeightInit, hidden.Activation), rng);
            layers.Add(dense);
            shape = dense.OutputShape;
            if (hidden.BatchNorm)
                layers.Add(new BatchNorm1DLayer(shape));
            layers.Add(new ActivationLayer(shape, hidden.Activation));
            layers.Add(new DropoutLayer(shape, hidden.Dropout));
        }

        var output = new DenseLayer(shape, spec.Labels.Count);
        output.Initialize(spec.WeightInit == WeightInitKind.Auto ? WeightInitKind.Xavier : spec.WeightInit, rng);
        layers.Add(output);

        return new NeuralNetwork(spec.InputShape, layers, spec.Labels, metadata);
    }

    private static WeightInitKind ResolveInit(WeightInitKind requested, ActivationKind activation)
    {
        if (requested != WeightInitKind.Auto)
            return requested;
        return activation is ActivationKind.Tanh or ActivationKind.Sigmoid ? WeightInitKind.Xavier : WeightInitKind.He;
    }

    /// <summary>Recovers the hidden-layer description from an existing network (for the designer).</summary>
    public static List<HiddenLayerSpec> DescribeHiddenLayers(NeuralNetwork network)
    {
        var result = new List<HiddenLayerSpec>();
        HiddenLayerSpec? current = null;
        var layers = network.Layers;
        for (int i = 0; i < layers.Count; i++)
        {
            switch (layers[i])
            {
                case DenseLayer dense when i < layers.Count - 1:
                    current = new HiddenLayerSpec { Units = dense.Units, Dropout = 0f };
                    result.Add(current);
                    break;
                case BatchNorm1DLayer when current is not null:
                    current.BatchNorm = true;
                    break;
                case ActivationLayer act when current is not null:
                    current.Activation = act.Activation;
                    break;
                case DropoutLayer drop when current is not null:
                    current.Dropout = drop.Rate;
                    break;
            }
        }
        return result;
    }
}
