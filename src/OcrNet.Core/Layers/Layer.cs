using OcrNet.Core.Tensors;

namespace OcrNet.Core.Layers;

/// <summary>
/// Stable layer identifiers used by the model file format. Never renumber existing values;
/// ids from 100 upwards are reserved for the convolutional layers of the planned v2.
/// </summary>
public enum LayerKind : ushort
{
    Dense = 1,
    Activation = 2,
    Dropout = 3,
    BatchNorm1D = 4,

    // Reserved for the CNN version (see README).
    Conv2D = 100,
    MaxPool2D = 101,
    AvgPool2D = 102,
    Flatten = 103,
    BatchNorm2D = 104,
    GlobalAvgPool = 105,
    Residual = 106,
}

/// <summary>
/// Base class for all layers. Layers process whole mini-batches, cache what they need for the
/// backward pass and reuse their output buffers between batches.
/// </summary>
public abstract class Layer
{
    private static readonly IReadOnlyList<Parameter> NoParameters = [];
    private static readonly IReadOnlyList<NamedArray> NoState = [];

    protected Layer(TensorShape inputShape, TensorShape outputShape)
    {
        InputShape = inputShape;
        OutputShape = outputShape;
    }

    public abstract LayerKind Kind { get; }

    /// <summary>Short human-readable description, e.g. "Dense 512".</summary>
    public abstract string DisplayName { get; }

    public TensorShape InputShape { get; }

    public TensorShape OutputShape { get; }

    public virtual IReadOnlyList<Parameter> Parameters => NoParameters;

    /// <summary>Non-trainable arrays that must be saved with the model (e.g. batch-norm running statistics).</summary>
    public virtual IReadOnlyList<NamedArray> State => NoState;

    /// <summary>Output of the most recent forward pass (used by the visualizer).</summary>
    public Tensor? LastOutput { get; protected set; }

    /// <summary>Approximate multiply-adds × 2 per sample for the forward pass.</summary>
    public virtual long FlopsPerSample => 0;

    public long ParameterCount => Parameters.Sum(p => (long)p.Length);

    public abstract Tensor Forward(Tensor input, bool training);

    /// <summary>
    /// Back-propagates <paramref name="gradOutput"/>, writing (not accumulating) parameter gradients.
    /// Returns the gradient with respect to the input, or <c>null</c> when <paramref name="needInputGrad"/> is false.
    /// </summary>
    public abstract Tensor? Backward(Tensor gradOutput, bool needInputGrad);

    /// <summary>Writes the layer's hyperparameters (not its weights) for the model file.</summary>
    public virtual void WriteConfig(BinaryWriter writer)
    {
    }
}
