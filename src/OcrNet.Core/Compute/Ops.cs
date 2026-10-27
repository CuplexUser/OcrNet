using System.Numerics;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Compute;

/// <summary>Small SIMD helpers on top of <see cref="TensorPrimitives"/>.</summary>
public static class Ops
{
    /// <summary>Adds <paramref name="bias"/> to every sample (row) of <paramref name="t"/>.</summary>
    public static void AddRowVector(Tensor t, ReadOnlySpan<float> bias)
    {
        for (int i = 0; i < t.Batch; i++)
        {
            var row = t.Sample(i);
            TensorPrimitives.Add(row, bias, row);
        }
    }

    /// <summary>Writes the column sums of <paramref name="t"/> (sum over the batch) into <paramref name="sums"/>.</summary>
    public static void SumRows(Tensor t, Span<float> sums)
    {
        sums.Clear();
        for (int i = 0; i < t.Batch; i++)
            TensorPrimitives.Add(sums, t.Sample(i), sums);
    }

    /// <summary>Numerically stable in-place softmax of one row.</summary>
    public static void SoftmaxInPlace(Span<float> row)
    {
        float max = TensorPrimitives.Max(row);
        TensorPrimitives.Subtract(row, max, row);
        TensorPrimitives.Exp(row, row);
        float sum = TensorPrimitives.Sum(row);
        TensorPrimitives.Multiply(row, 1f / sum, row);
    }

    /// <summary>Runs <paramref name="body"/> over [0, length) in chunks, in parallel when the range is large.</summary>
    public static void ForChunks(int length, Action<int, int> body, int chunk = 32 * 1024)
    {
        if (length <= chunk * 2 || Parallelism.MaxThreads == 1)
        {
            body(0, length);
            return;
        }
        int chunks = (length + chunk - 1) / chunk;
        Parallel.For(0, chunks, Parallelism.Options, c =>
        {
            int start = c * chunk;
            body(start, Math.Min(chunk, length - start));
        });
    }

    /// <summary>dst[i] = cond[i] &gt; 0 ? src[i] : src[i] * negativeScale.</summary>
    public static void MaskByPositive(ReadOnlySpan<float> cond, ReadOnlySpan<float> src, Span<float> dst, float negativeScale)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var zero = Vector<float>.Zero;
            var scale = new Vector<float>(negativeScale);
            int w = Vector<float>.Count;
            for (; i <= cond.Length - w; i += w)
            {
                var c = new Vector<float>(cond.Slice(i));
                var s = new Vector<float>(src.Slice(i));
                Vector.ConditionalSelect(Vector.GreaterThan(c, zero), s, s * scale).CopyTo(dst.Slice(i));
            }
        }
        for (; i < cond.Length; i++)
            dst[i] = cond[i] > 0f ? src[i] : src[i] * negativeScale;
    }
}
