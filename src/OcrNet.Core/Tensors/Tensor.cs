namespace OcrNet.Core.Tensors;

/// <summary>
/// A mini-batch of samples stored contiguously in NCHW order. The backing array can be larger
/// than the batch currently held so buffers are reused across batches of different sizes
/// (e.g. the smaller last batch of an epoch) without reallocating.
/// </summary>
public sealed class Tensor
{
    private float[] _data;

    public Tensor(int batch, TensorShape shape)
    {
        Shape = shape;
        Batch = batch;
        _data = new float[Math.Max(1, batch) * shape.Size];
    }

    public Tensor(int batch, TensorShape shape, float[] data)
    {
        if (data.Length < batch * shape.Size)
            throw new ArgumentException("Data array is smaller than batch × shape.", nameof(data));
        Shape = shape;
        Batch = batch;
        _data = data;
    }

    public TensorShape Shape { get; }

    public int Batch { get; private set; }

    /// <summary>Number of floats per sample.</summary>
    public int SampleSize => Shape.Size;

    /// <summary>Number of valid floats (Batch × SampleSize).</summary>
    public int Length => Batch * Shape.Size;

    /// <summary>The raw backing array; only the first <see cref="Length"/> elements are valid.</summary>
    public float[] Data => _data;

    public Span<float> Span => _data.AsSpan(0, Length);

    public Span<float> Sample(int index) => _data.AsSpan(index * Shape.Size, Shape.Size);

    /// <summary>Changes the batch size, growing the backing array if needed. Contents are undefined afterwards.</summary>
    public void Resize(int batch)
    {
        int needed = batch * Shape.Size;
        if (_data.Length < needed)
            _data = new float[needed];
        Batch = batch;
    }

    /// <summary>Returns <paramref name="tensor"/> resized to <paramref name="batch"/>, allocating it on first use.</summary>
    public static Tensor Ensure(ref Tensor? tensor, int batch, TensorShape shape)
    {
        if (tensor is null || tensor.Shape != shape)
            tensor = new Tensor(batch, shape);
        else
            tensor.Resize(batch);
        return tensor;
    }

    public void Clear() => Span.Clear();
}
