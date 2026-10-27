using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;
using OcrNet.Core.Compute;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Layers;

/// <summary>
/// Inverted dropout: during training each activation is zeroed with probability <see cref="Rate"/>
/// and the survivors are scaled by 1/(1−rate), so inference is a no-op. A rate of 0 is a pass-through.
/// </summary>
public sealed class DropoutLayer : Layer
{
    private float[] _mask = [];
    private Tensor? _output;
    private Tensor? _gradInput;
    private bool _maskActive;
    private FastRandom _rng = new(1);
    private float _rate;

    public DropoutLayer(TensorShape shape, float rate)
        : base(shape, shape)
    {
        Rate = rate;
    }

    public float Rate
    {
        get => _rate;
        set => _rate = Math.Clamp(value, 0f, 0.95f);
    }

    public override LayerKind Kind => LayerKind.Dropout;

    public override string DisplayName => $"Dropout {Rate:0.##}";

    public void Seed(ulong seed) => _rng = new FastRandom(seed);

    public override Tensor Forward(Tensor input, bool training)
    {
        _maskActive = training && Rate > 0f;
        if (!_maskActive)
        {
            LastOutput = input;
            return input;
        }

        int length = input.Length;
        if (_mask.Length < length)
            _mask = new float[length];
        float keep = 1f - Rate;
        float scale = 1f / keep;
        for (int i = 0; i < length; i++)
            _mask[i] = _rng.NextFloat() < keep ? scale : 0f;

        var output = Tensor.Ensure(ref _output, input.Batch, OutputShape);
        TensorPrimitives.Multiply(input.Span, _mask.AsSpan(0, length), output.Span);
        LastOutput = output;
        return output;
    }

    public override Tensor? Backward(Tensor gradOutput, bool needInputGrad)
    {
        if (!needInputGrad)
            return null;
        if (!_maskActive)
            return gradOutput;
        var gradInput = Tensor.Ensure(ref _gradInput, gradOutput.Batch, InputShape);
        TensorPrimitives.Multiply(gradOutput.Span, _mask.AsSpan(0, gradOutput.Length), gradInput.Span);
        return gradInput;
    }

    public override void WriteConfig(BinaryWriter writer) => writer.Write(Rate);

    internal static DropoutLayer ReadConfig(TensorShape inputShape, BinaryReader reader) => new(inputShape, reader.ReadSingle());
}
