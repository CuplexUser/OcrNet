namespace OcrNet.Core.Layers;

/// <summary>A named array of floats stored with a model (weights, biases, running statistics).</summary>
public class NamedArray(string name, int[] dims)
{
    public string Name { get; } = name;

    /// <summary>Logical dimensions, e.g. [out, in] for a dense weight matrix.</summary>
    public int[] Dims { get; } = dims;

    public float[] Value { get; } = new float[dims.Aggregate(1, (a, b) => a * b)];

    public int Length => Value.Length;
}

/// <summary>A trainable tensor together with its gradient.</summary>
public sealed class Parameter(string name, int[] dims, bool applyWeightDecay) : NamedArray(name, dims)
{
    public float[] Grad { get; } = new float[dims.Aggregate(1, (a, b) => a * b)];

    /// <summary>Weights get L2 / decoupled weight decay; biases and normalization parameters do not.</summary>
    public bool ApplyWeightDecay { get; } = applyWeightDecay;
}
