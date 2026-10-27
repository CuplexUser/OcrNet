using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;
using OcrNet.Core.Compute;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Layers;

/// <summary>Stable ids (stored in model files).</summary>
public enum ActivationKind : byte
{
    ReLU = 1,
    LeakyReLU = 2,
    Tanh = 3,
    Sigmoid = 4,
    Gelu = 5,
}

/// <summary>Element-wise non-linearity.</summary>
public sealed class ActivationLayer : Layer
{
    public const float LeakySlope = 0.01f;

    private Tensor? _input;
    private Tensor? _output;
    private Tensor? _gradInput;

    public ActivationLayer(TensorShape shape, ActivationKind activation)
        : base(shape, shape)
    {
        Activation = activation;
    }

    public ActivationKind Activation { get; }

    public override LayerKind Kind => LayerKind.Activation;

    public override string DisplayName => Activation.ToString();

    public override Tensor Forward(Tensor input, bool training)
    {
        _input = input;
        var output = Tensor.Ensure(ref _output, input.Batch, OutputShape);
        ReadOnlySpan<float> x = input.Span;
        Span<float> y = output.Span;
        switch (Activation)
        {
            case ActivationKind.ReLU:
                TensorPrimitives.Max(x, 0f, y);
                break;
            case ActivationKind.LeakyReLU:
                Ops.MaskByPositive(x, x, y, LeakySlope);
                break;
            case ActivationKind.Tanh:
                TensorPrimitives.Tanh(x, y);
                break;
            case ActivationKind.Sigmoid:
                TensorPrimitives.Sigmoid(x, y);
                break;
            case ActivationKind.Gelu:
                for (int i = 0; i < x.Length; i++)
                    y[i] = Gelu(x[i]);
                break;
        }
        LastOutput = output;
        return output;
    }

    public override Tensor? Backward(Tensor gradOutput, bool needInputGrad)
    {
        if (!needInputGrad)
            return null;
        var input = _input ?? throw new InvalidOperationException("Backward called before Forward.");
        var gradInput = Tensor.Ensure(ref _gradInput, gradOutput.Batch, InputShape);
        ReadOnlySpan<float> x = input.Span;
        ReadOnlySpan<float> y = _output!.Span;
        ReadOnlySpan<float> dy = gradOutput.Span;
        Span<float> dx = gradInput.Span;
        switch (Activation)
        {
            case ActivationKind.ReLU:
                Ops.MaskByPositive(x, dy, dx, 0f);
                break;
            case ActivationKind.LeakyReLU:
                Ops.MaskByPositive(x, dy, dx, LeakySlope);
                break;
            case ActivationKind.Tanh:
                // dx = dy · (1 − y²)
                TensorPrimitives.Multiply(y, y, dx);
                TensorPrimitives.Subtract(1f, dx, dx);
                TensorPrimitives.Multiply(dx, dy, dx);
                break;
            case ActivationKind.Sigmoid:
                // dx = dy · y · (1 − y)
                TensorPrimitives.Subtract(1f, y, dx);
                TensorPrimitives.Multiply(dx, y, dx);
                TensorPrimitives.Multiply(dx, dy, dx);
                break;
            case ActivationKind.Gelu:
                for (int i = 0; i < x.Length; i++)
                    dx[i] = dy[i] * GeluDerivative(x[i]);
                break;
        }
        return gradInput;
    }

    // GELU, tanh approximation.
    private const float GeluC = 0.7978845608f; // √(2/π)
    private const float GeluA = 0.044715f;

    private static float Gelu(float x) => 0.5f * x * (1f + MathF.Tanh(GeluC * (x + GeluA * x * x * x)));

    private static float GeluDerivative(float x)
    {
        float u = GeluC * (x + GeluA * x * x * x);
        float t = MathF.Tanh(u);
        float du = GeluC * (1f + 3f * GeluA * x * x);
        return 0.5f * (1f + t) + 0.5f * x * (1f - t * t) * du;
    }

    public override void WriteConfig(BinaryWriter writer) => writer.Write((byte)Activation);

    internal static ActivationLayer ReadConfig(TensorShape inputShape, BinaryReader reader)
        => new(inputShape, (ActivationKind)reader.ReadByte());
}
