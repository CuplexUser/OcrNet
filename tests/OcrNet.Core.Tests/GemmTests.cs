using OcrNet.Core.Compute;

namespace OcrNet.Core.Tests;

public class GemmTests
{
    public static TheoryData<bool, bool, int, int, int> Shapes()
    {
        var data = new TheoryData<bool, bool, int, int, int>();
        int[][] sizes = [[1, 1, 1], [5, 7, 3], [6, 16, 8], [13, 33, 300], [64, 130, 257], [128, 62, 512], [1, 512, 784]];
        foreach (var s in sizes)
            foreach (bool ta in new[] { false, true })
                foreach (bool tb in new[] { false, true })
                    data.Add(ta, tb, s[0], s[1], s[2]);
        return data;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Multiply_MatchesNaiveReference(bool transA, bool transB, int m, int n, int k)
    {
        var rng = new FastRandom(123);
        int lda = transA ? m : k;
        int ldb = transB ? k : n;
        var a = Random(rng, m * k);
        var b = Random(rng, k * n);
        var expected = Random(rng, m * n);
        var actual = (float[])expected.Clone();

        Gemm.MultiplyNaive(transA, transB, m, n, k, a, lda, b, ldb, expected, n, accumulate: true);
        Gemm.Multiply(transA, transB, m, n, k, a, lda, b, ldb, actual, n, accumulate: true);

        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-3f * (1 + MathF.Abs(expected[i])),
                $"Mismatch at {i}: {expected[i]} vs {actual[i]}");
    }

    [Fact]
    public void Multiply_WithoutAccumulate_OverwritesOutput()
    {
        var a = new float[] { 1, 2, 3, 4 };      // 2×2
        var b = new float[] { 5, 6, 7, 8 };      // 2×2
        var c = new float[] { 99, 99, 99, 99 };
        Gemm.Multiply(false, false, 2, 2, 2, a, 2, b, 2, c, 2);
        Assert.Equal([19f, 22f, 43f, 50f], c);
    }

    [Fact]
    public void Multiply_SingleThreaded_GivesSameResult()
    {
        var rng = new FastRandom(5);
        var a = Random(rng, 100 * 200);
        var b = Random(rng, 200 * 150);
        var parallel = new float[100 * 150];
        var serial = new float[100 * 150];
        Gemm.Multiply(false, false, 100, 150, 200, a, 200, b, 150, parallel, 150);
        int previous = Parallelism.MaxThreads;
        try
        {
            Parallelism.MaxThreads = 1;
            Gemm.Multiply(false, false, 100, 150, 200, a, 200, b, 150, serial, 150);
        }
        finally
        {
            Parallelism.MaxThreads = previous;
        }
        Assert.Equal(serial, parallel);
    }

    internal static float[] Random(FastRandom rng, int length)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = rng.NextFloat(-1f, 1f);
        return data;
    }
}
