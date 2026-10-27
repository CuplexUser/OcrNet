using OcrNet.Core.Tensors;

namespace OcrNet.Core.Layers;

/// <summary>Batch normalization over the feature dimension of a flat activation vector.</summary>
public sealed class BatchNorm1DLayer : Layer
{
    private readonly Parameter _gamma;
    private readonly Parameter _beta;
    private readonly NamedArray _runningMean;
    private readonly NamedArray _runningVar;
    private readonly float[] _mean;
    private readonly float[] _invStd;
    private readonly float[] _sumDy;
    private readonly float[] _sumDyXhat;
    private Tensor? _xhat;
    private Tensor? _output;
    private Tensor? _gradInput;
    private bool _usedRunningStats;

    public BatchNorm1DLayer(TensorShape shape, float momentum = 0.1f, float epsilon = 1e-5f)
        : base(shape, shape)
    {
        Features = shape.Size;
        Momentum = momentum;
        Epsilon = epsilon;
        _gamma = new Parameter("gamma", [Features], applyWeightDecay: false);
        _beta = new Parameter("beta", [Features], applyWeightDecay: false);
        _runningMean = new NamedArray("running_mean", [Features]);
        _runningVar = new NamedArray("running_var", [Features]);
        Array.Fill(_gamma.Value, 1f);
        Array.Fill(_runningVar.Value, 1f);
        _mean = new float[Features];
        _invStd = new float[Features];
        _sumDy = new float[Features];
        _sumDyXhat = new float[Features];
        Parameters = [_gamma, _beta];
        State = [_runningMean, _runningVar];
    }

    public int Features { get; }

    public float Momentum { get; }

    public float Epsilon { get; }

    public override LayerKind Kind => LayerKind.BatchNorm1D;

    public override string DisplayName => "BatchNorm";

    public override IReadOnlyList<Parameter> Parameters { get; }

    public override IReadOnlyList<NamedArray> State { get; }

    public override long FlopsPerSample => 4L * Features;

    public override Tensor Forward(Tensor input, bool training)
    {
        int n = input.Batch;
        int f = Features;
        var output = Tensor.Ensure(ref _output, n, OutputShape);
        float[] x = input.Data, y = output.Data, gamma = _gamma.Value, beta = _beta.Value;

        _usedRunningStats = !(training && n > 1);
        if (!_usedRunningStats)
        {
            var xhat = Tensor.Ensure(ref _xhat, n, OutputShape).Data;
            Array.Clear(_mean);
            Array.Clear(_invStd); // used as the variance accumulator first
            for (int i = 0; i < n; i++)
                for (int j = 0; j < f; j++)
                    _mean[j] += x[i * f + j];
            for (int j = 0; j < f; j++)
                _mean[j] /= n;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < f; j++)
                {
                    float d = x[i * f + j] - _mean[j];
                    _invStd[j] += d * d;
                }
            float[] rm = _runningMean.Value, rv = _runningVar.Value;
            for (int j = 0; j < f; j++)
            {
                float variance = _invStd[j] / n;
                rm[j] = (1 - Momentum) * rm[j] + Momentum * _mean[j];
                rv[j] = (1 - Momentum) * rv[j] + Momentum * variance * n / (n - 1);
                _invStd[j] = 1f / MathF.Sqrt(variance + Epsilon);
            }
            for (int i = 0; i < n; i++)
                for (int j = 0; j < f; j++)
                {
                    int k = i * f + j;
                    float h = (x[k] - _mean[j]) * _invStd[j];
                    xhat[k] = h;
                    y[k] = gamma[j] * h + beta[j];
                }
        }
        else
        {
            // Inference, or a training batch of one sample where batch statistics are undefined.
            var xhat = Tensor.Ensure(ref _xhat, n, OutputShape).Data;
            float[] rm = _runningMean.Value, rv = _runningVar.Value;
            for (int j = 0; j < f; j++)
                _invStd[j] = 1f / MathF.Sqrt(rv[j] + Epsilon);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < f; j++)
                {
                    int k = i * f + j;
                    float h = (x[k] - rm[j]) * _invStd[j];
                    xhat[k] = h;
                    y[k] = gamma[j] * h + beta[j];
                }
        }
        LastOutput = output;
        return output;
    }

    public override Tensor? Backward(Tensor gradOutput, bool needInputGrad)
    {
        int n = gradOutput.Batch;
        int f = Features;
        float[] dy = gradOutput.Data, xhat = _xhat!.Data, gamma = _gamma.Value;
        Array.Clear(_sumDy);
        Array.Clear(_sumDyXhat);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < f; j++)
            {
                int k = i * f + j;
                _sumDy[j] += dy[k];
                _sumDyXhat[j] += dy[k] * xhat[k];
            }
        Array.Copy(_sumDyXhat, _gamma.Grad, f);
        Array.Copy(_sumDy, _beta.Grad, f);

        if (!needInputGrad)
            return null;

        // dx = γ·invStd/N · (N·dy − Σdy − x̂·Σ(dy·x̂))
        var gradInput = Tensor.Ensure(ref _gradInput, n, InputShape);
        float[] dx = gradInput.Data;
        if (_usedRunningStats)
        {
            // Statistics were constants, so the layer is a per-feature affine map.
            for (int i = 0; i < n; i++)
                for (int j = 0; j < f; j++)
                    dx[i * f + j] = dy[i * f + j] * gamma[j] * _invStd[j];
            return gradInput;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < f; j++)
            {
                int k = i * f + j;
                dx[k] = gamma[j] * _invStd[j] / n * (n * dy[k] - _sumDy[j] - xhat[k] * _sumDyXhat[j]);
            }
        return gradInput;
    }

    public override void WriteConfig(BinaryWriter writer)
    {
        writer.Write(Momentum);
        writer.Write(Epsilon);
    }

    internal static BatchNorm1DLayer ReadConfig(TensorShape inputShape, BinaryReader reader)
        => new(inputShape, reader.ReadSingle(), reader.ReadSingle());
}
