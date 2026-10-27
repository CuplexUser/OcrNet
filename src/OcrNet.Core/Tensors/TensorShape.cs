namespace OcrNet.Core.Tensors;

/// <summary>
/// Shape of a single sample in channel/height/width (CHW) order. The batch dimension is
/// tracked separately by <see cref="Tensor"/>. A flat feature vector is (C = n, H = 1, W = 1).
/// </summary>
public readonly record struct TensorShape(int C, int H, int W)
{
    public int Size => C * H * W;

    public static TensorShape Vector(int length) => new(length, 1, 1);

    public bool IsVector => H == 1 && W == 1;

    public override string ToString() => IsVector ? $"{C}" : $"{C}×{H}×{W}";
}
