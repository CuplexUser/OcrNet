using OcrNet.Core.Compute;
using OcrNet.Core.Layers;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Network;

/// <summary>
/// A feed-forward stack of layers ending in class logits. Training and inference share the
/// layer buffers, so every pass must hold <see cref="SyncRoot"/>; the trainer takes it once per
/// mini-batch, which lets the UI run predictions while a model is being trained.
/// </summary>
public sealed class NeuralNetwork
{
    private readonly List<Layer> _layers;
    private Tensor? _predictInput;
    private Tensor? _predictProbabilities;

    public NeuralNetwork(TensorShape inputShape, IEnumerable<Layer> layers, IReadOnlyList<string> labels, ModelMetadata? metadata = null)
    {
        InputShape = inputShape;
        _layers = [.. layers];
        Labels = [.. labels];
        Metadata = metadata ?? new ModelMetadata();

        var shape = inputShape;
        foreach (var layer in _layers)
        {
            if (layer.InputShape.Size != shape.Size)
                throw new ArgumentException($"Layer {layer.DisplayName} expects {layer.InputShape} but receives {shape}.");
            shape = layer.OutputShape;
        }
        if (shape.Size != Labels.Count)
            throw new ArgumentException($"Network outputs {shape.Size} values but has {Labels.Count} class labels.");
    }

    public object SyncRoot { get; } = new();

    public TensorShape InputShape { get; }

    public IReadOnlyList<Layer> Layers => _layers;

    /// <summary>Class labels in output order. A label may contain several characters (merged classes, e.g. "Cc").</summary>
    public IReadOnlyList<string> Labels { get; }

    public int ClassCount => Labels.Count;

    public ModelMetadata Metadata { get; set; }

    public IEnumerable<Parameter> Parameters => _layers.SelectMany(l => l.Parameters);

    public long ParameterCount => _layers.Sum(l => l.ParameterCount);

    public long FlopsPerSample => _layers.Sum(l => l.FlopsPerSample);

    /// <summary>Forward pass returning logits. Callers must hold <see cref="SyncRoot"/>.</summary>
    public Tensor Forward(Tensor input, bool training)
    {
        var x = input;
        foreach (var layer in _layers)
            x = layer.Forward(x, training);
        return x;
    }

    /// <summary>Backward pass from dLoss/dLogits. Callers must hold <see cref="SyncRoot"/>.</summary>
    public void Backward(Tensor gradLogits)
    {
        var g = gradLogits;
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            var next = _layers[i].Backward(g, needInputGrad: i > 0);
            if (next is null)
                break;
            g = next;
        }
    }

    /// <summary>Classifies one preprocessed sample and returns the class probabilities.</summary>
    public float[] Predict(ReadOnlySpan<float> input)
    {
        if (input.Length != InputShape.Size)
            throw new ArgumentException($"Expected {InputShape.Size} inputs, got {input.Length}.", nameof(input));
        lock (SyncRoot)
        {
            var x = Tensor.Ensure(ref _predictInput, 1, InputShape);
            input.CopyTo(x.Span);
            var logits = Forward(x, training: false);
            var probs = Tensor.Ensure(ref _predictProbabilities, 1, logits.Shape);
            logits.Span.CopyTo(probs.Span);
            Ops.SoftmaxInPlace(probs.Span);
            return probs.Span.ToArray();
        }
    }

    /// <summary>
    /// Like <see cref="Predict"/>, but also returns a copy of every layer's output for this sample
    /// (used by the visualizer to show activations).
    /// </summary>
    public (float[] Probabilities, float[][] LayerOutputs) PredictDetailed(ReadOnlySpan<float> input)
    {
        lock (SyncRoot)
        {
            var probabilities = Predict(input);
            var outputs = _layers.Select(l => l.LastOutput is { } o ? o.Sample(0).ToArray() : []).ToArray();
            outputs[^1] = probabilities; // report the softmax output rather than raw logits
            return (probabilities, outputs);
        }
    }

    /// <summary>Copies all trainable parameters and state arrays (used for best-epoch checkpoints).</summary>
    public float[][] SnapshotWeights()
    {
        lock (SyncRoot)
            return [.. AllArrays().Select(a => (float[])a.Value.Clone())];
    }

    public void RestoreWeights(float[][] snapshot)
    {
        lock (SyncRoot)
        {
            int i = 0;
            foreach (var a in AllArrays())
                snapshot[i++].CopyTo(a.Value, 0);
        }
    }

    private IEnumerable<NamedArray> AllArrays() =>
        _layers.SelectMany(l => l.Parameters.Cast<NamedArray>().Concat(l.State));

    public override string ToString() =>
        string.Join(" → ", [InputShape.ToString(), .. _layers.Select(l => l.DisplayName)]);
}
