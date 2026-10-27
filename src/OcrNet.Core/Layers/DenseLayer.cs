using OcrNet.Core.Compute;
using OcrNet.Core.Init;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Layers;

/// <summary>Fully connected layer: <c>Y = X·Wᵀ + b</c> with W stored as [units, inputs].</summary>
public sealed class DenseLayer : Layer
{
    private readonly Parameter _weights;
    private readonly Parameter _bias;
    private Tensor? _input;
    private Tensor? _output;
    private Tensor? _gradInput;

    public DenseLayer(TensorShape inputShape, int units)
        : base(inputShape, TensorShape.Vector(units))
    {
        Inputs = inputShape.Size;
        Units = units;
        _weights = new Parameter("weights", [units, Inputs], applyWeightDecay: true);
        _bias = new Parameter("bias", [units], applyWeightDecay: false);
        Parameters = [_weights, _bias];
    }

    public int Inputs { get; }

    public int Units { get; }

    /// <summary>Row-major [Units, Inputs].</summary>
    public float[] Weights => _weights.Value;

    public float[] Bias => _bias.Value;

    public override LayerKind Kind => LayerKind.Dense;

    public override string DisplayName => $"Dense {Units}";

    public override IReadOnlyList<Parameter> Parameters { get; }

    public override long FlopsPerSample => 2L * Inputs * Units;

    public void Initialize(WeightInitKind kind, FastRandom rng)
    {
        WeightInit.Fill(_weights.Value, Inputs, Units, kind, rng);
        Array.Clear(_bias.Value);
    }

    public override Tensor Forward(Tensor input, bool training)
    {
        _input = input;
        var output = Tensor.Ensure(ref _output, input.Batch, OutputShape);
        Gemm.Multiply(false, true, input.Batch, Units, Inputs, input.Data, Inputs, _weights.Value, Inputs, output.Data, Units);
        Ops.AddRowVector(output, _bias.Value);
        LastOutput = output;
        return output;
    }

    public override Tensor? Backward(Tensor gradOutput, bool needInputGrad)
    {
        var input = _input ?? throw new InvalidOperationException("Backward called before Forward.");
        int n = gradOutput.Batch;

        // dW = dYᵀ·X, db = Σ dY
        Gemm.Multiply(true, false, Units, Inputs, n, gradOutput.Data, Units, input.Data, Inputs, _weights.Grad, Inputs);
        Ops.SumRows(gradOutput, _bias.Grad);

        if (!needInputGrad)
            return null;

        // dX = dY·W
        var gradInput = Tensor.Ensure(ref _gradInput, n, InputShape);
        Gemm.Multiply(false, false, n, Inputs, Units, gradOutput.Data, Units, _weights.Value, Inputs, gradInput.Data, Inputs);
        return gradInput;
    }

    public override void WriteConfig(BinaryWriter writer) => writer.Write(Units);

    internal static DenseLayer ReadConfig(TensorShape inputShape, BinaryReader reader) => new(inputShape, reader.ReadInt32());
}
