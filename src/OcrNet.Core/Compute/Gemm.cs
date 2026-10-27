using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OcrNet.Core.Compute;

/// <summary>
/// Single-precision matrix multiply <c>C = op(A) · op(B)</c> (or <c>C += …</c>) for row-major
/// matrices. Both operands are packed into cache-friendly panels per K block, then a 6×16
/// register-blocked AVX2/FMA micro-kernel computes the output in tiles that are distributed
/// across threads with <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/>.
/// </summary>
public static unsafe class Gemm
{
    private const int MR = 6;          // micro-kernel rows
    private const int NR = 16;         // micro-kernel columns (two 256-bit vectors)
    private const int KC = 256;        // K block: one A panel (6 KB) + one B panel (16 KB) stay in L1/L2
    private const int TileMPanels = 4; // a work item covers 4×6 = 24 rows …
    private const int TileNPanels = 8; // … and 8×16 = 128 columns of C
    private const long ParallelThreshold = 1L << 18; // multiply-adds per K block before going parallel

    /// <summary>
    /// Computes <c>C = op(A)·op(B)</c>, or <c>C += op(A)·op(B)</c> when <paramref name="accumulate"/> is set.
    /// op(A) is m×k, op(B) is k×n and C is m×n; <c>lda/ldb/ldc</c> are the row strides of the stored arrays.
    /// </summary>
    public static void Multiply(
        bool transA, bool transB, int m, int n, int k,
        float[] a, int lda, float[] b, int ldb, float[] c, int ldc, bool accumulate = false)
    {
        if (m <= 0 || n <= 0)
            return;
        if (k <= 0)
        {
            if (!accumulate)
                for (int i = 0; i < m; i++)
                    c.AsSpan(i * ldc, n).Clear();
            return;
        }

        int panelsM = (m + MR - 1) / MR;
        int panelsN = (n + NR - 1) / NR;
        int kcMax = Math.Min(KC, k);
        float[] packA = ArrayPool<float>.Shared.Rent(panelsM * MR * kcMax);
        float[] packB = ArrayPool<float>.Shared.Rent(panelsN * NR * kcMax);
        try
        {
            int tilesM = (panelsM + TileMPanels - 1) / TileMPanels;
            int tilesN = (panelsN + TileNPanels - 1) / TileNPanels;
            for (int p0 = 0; p0 < k; p0 += KC)
            {
                int kc = Math.Min(KC, k - p0);
                bool add = accumulate || p0 > 0;
                bool parallel = (long)m * n * kc >= ParallelThreshold && Parallelism.MaxThreads > 1;

                if (parallel)
                {
                    var options = Parallelism.Options;
                    Parallel.For(0, panelsM, options, ip => PackA(transA, a, lda, m, p0, kc, ip, packA));
                    Parallel.For(0, panelsN, options, jp => PackB(transB, b, ldb, n, p0, kc, jp, packB));
                    Parallel.For(0, tilesM * tilesN, options,
                        t => ComputeTile(t / tilesN, t % tilesN, panelsM, panelsN, m, n, kc, packA, packB, c, ldc, add));
                }
                else
                {
                    for (int ip = 0; ip < panelsM; ip++) PackA(transA, a, lda, m, p0, kc, ip, packA);
                    for (int jp = 0; jp < panelsN; jp++) PackB(transB, b, ldb, n, p0, kc, jp, packB);
                    for (int t = 0; t < tilesM * tilesN; t++)
                        ComputeTile(t / tilesN, t % tilesN, panelsM, panelsN, m, n, kc, packA, packB, c, ldc, add);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packA);
            ArrayPool<float>.Shared.Return(packB);
        }
    }

    /// <summary>Packs rows [ip·MR, ip·MR+MR) × cols [p0, p0+kc) of op(A) as kc groups of MR values (zero padded).</summary>
    private static void PackA(bool transA, float[] a, int lda, int m, int p0, int kc, int ip, float[] pack)
    {
        int i0 = ip * MR;
        int rows = Math.Min(MR, m - i0);
        int dst = ip * MR * kc;
        if (!transA)
        {
            for (int p = 0; p < kc; p++)
            {
                int src = i0 * lda + p0 + p;
                int r = 0;
                for (; r < rows; r++, src += lda)
                    pack[dst + r] = a[src];
                for (; r < MR; r++)
                    pack[dst + r] = 0f;
                dst += MR;
            }
        }
        else
        {
            for (int p = 0; p < kc; p++)
            {
                int src = (p0 + p) * lda + i0;
                int r = 0;
                for (; r < rows; r++)
                    pack[dst + r] = a[src + r];
                for (; r < MR; r++)
                    pack[dst + r] = 0f;
                dst += MR;
            }
        }
    }

    /// <summary>Packs rows [p0, p0+kc) × cols [jp·NR, jp·NR+NR) of op(B) as kc groups of NR values (zero padded).</summary>
    private static void PackB(bool transB, float[] b, int ldb, int n, int p0, int kc, int jp, float[] pack)
    {
        int j0 = jp * NR;
        int cols = Math.Min(NR, n - j0);
        int dst = jp * NR * kc;
        if (!transB)
        {
            for (int p = 0; p < kc; p++)
            {
                var row = b.AsSpan((p0 + p) * ldb + j0, cols);
                row.CopyTo(pack.AsSpan(dst, cols));
                if (cols < NR)
                    pack.AsSpan(dst + cols, NR - cols).Clear();
                dst += NR;
            }
        }
        else
        {
            for (int p = 0; p < kc; p++)
            {
                int src = j0 * ldb + p0 + p;
                int col = 0;
                for (; col < cols; col++, src += ldb)
                    pack[dst + col] = b[src];
                for (; col < NR; col++)
                    pack[dst + col] = 0f;
                dst += NR;
            }
        }
    }

    private static void ComputeTile(
        int tileM, int tileN, int panelsM, int panelsN, int m, int n, int kc,
        float[] packA, float[] packB, float[] c, int ldc, bool add)
    {
        int ipEnd = Math.Min(panelsM, (tileM + 1) * TileMPanels);
        int jpEnd = Math.Min(panelsN, (tileN + 1) * TileNPanels);
        fixed (float* pA = packA, pB = packB, pC = c)
        {
            for (int jp = tileN * TileNPanels; jp < jpEnd; jp++)
            {
                int j0 = jp * NR;
                int cols = Math.Min(NR, n - j0);
                float* bPanel = pB + (long)jp * NR * kc;
                for (int ip = tileM * TileMPanels; ip < ipEnd; ip++)
                {
                    int i0 = ip * MR;
                    int rows = Math.Min(MR, m - i0);
                    float* aPanel = pA + (long)ip * MR * kc;
                    float* cBlock = pC + (long)i0 * ldc + j0;
                    if (Vector256.IsHardwareAccelerated)
                        Kernel6x16(aPanel, bPanel, kc, cBlock, ldc, rows, cols, add);
                    else
                        KernelScalar(aPanel, bPanel, kc, cBlock, ldc, rows, cols, add);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> MulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> acc)
        => Fma.IsSupported ? Fma.MultiplyAdd(a, b, acc) : acc + a * b;

    private static void Kernel6x16(float* a, float* b, int kc, float* c, int ldc, int rows, int cols, bool add)
    {
        Vector256<float> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
        Vector256<float> c30 = default, c31 = default, c40 = default, c41 = default, c50 = default, c51 = default;

        for (int p = 0; p < kc; p++)
        {
            Vector256<float> b0 = Vector256.Load(b);
            Vector256<float> b1 = Vector256.Load(b + 8);
            Vector256<float> av = Vector256.Create(a[0]);
            c00 = MulAdd(av, b0, c00); c01 = MulAdd(av, b1, c01);
            av = Vector256.Create(a[1]);
            c10 = MulAdd(av, b0, c10); c11 = MulAdd(av, b1, c11);
            av = Vector256.Create(a[2]);
            c20 = MulAdd(av, b0, c20); c21 = MulAdd(av, b1, c21);
            av = Vector256.Create(a[3]);
            c30 = MulAdd(av, b0, c30); c31 = MulAdd(av, b1, c31);
            av = Vector256.Create(a[4]);
            c40 = MulAdd(av, b0, c40); c41 = MulAdd(av, b1, c41);
            av = Vector256.Create(a[5]);
            c50 = MulAdd(av, b0, c50); c51 = MulAdd(av, b1, c51);
            a += MR;
            b += NR;
        }

        if (rows == MR && cols == NR)
        {
            StoreRow(c, c00, c01, add); c += ldc;
            StoreRow(c, c10, c11, add); c += ldc;
            StoreRow(c, c20, c21, add); c += ldc;
            StoreRow(c, c30, c31, add); c += ldc;
            StoreRow(c, c40, c41, add); c += ldc;
            StoreRow(c, c50, c51, add);
            return;
        }

        // Edge block: spill to a temporary and copy only the valid part.
        float* tmp = stackalloc float[MR * NR];
        c00.Store(tmp); c01.Store(tmp + 8);
        c10.Store(tmp + 16); c11.Store(tmp + 24);
        c20.Store(tmp + 32); c21.Store(tmp + 40);
        c30.Store(tmp + 48); c31.Store(tmp + 56);
        c40.Store(tmp + 64); c41.Store(tmp + 72);
        c50.Store(tmp + 80); c51.Store(tmp + 88);
        for (int r = 0; r < rows; r++)
        {
            float* src = tmp + r * NR;
            float* dst = c + (long)r * ldc;
            if (add)
                for (int j = 0; j < cols; j++) dst[j] += src[j];
            else
                for (int j = 0; j < cols; j++) dst[j] = src[j];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreRow(float* c, Vector256<float> lo, Vector256<float> hi, bool add)
    {
        if (add)
        {
            lo += Vector256.Load(c);
            hi += Vector256.Load(c + 8);
        }
        lo.Store(c);
        hi.Store(c + 8);
    }

    private static void KernelScalar(float* a, float* b, int kc, float* c, int ldc, int rows, int cols, bool add)
    {
        float* acc = stackalloc float[MR * NR];
        new Span<float>(acc, MR * NR).Clear();
        for (int p = 0; p < kc; p++)
        {
            for (int r = 0; r < MR; r++)
            {
                float av = a[r];
                float* row = acc + r * NR;
                for (int j = 0; j < NR; j++)
                    row[j] += av * b[j];
            }
            a += MR;
            b += NR;
        }
        for (int r = 0; r < rows; r++)
        {
            float* dst = c + (long)r * ldc;
            float* src = acc + r * NR;
            for (int j = 0; j < cols; j++)
                dst[j] = add ? dst[j] + src[j] : src[j];
        }
    }

    /// <summary>Straightforward triple loop; used by tests as the reference implementation.</summary>
    public static void MultiplyNaive(
        bool transA, bool transB, int m, int n, int k,
        float[] a, int lda, float[] b, int ldb, float[] c, int ldc, bool accumulate = false)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    float av = transA ? a[p * lda + i] : a[i * lda + p];
                    float bv = transB ? b[j * ldb + p] : b[p * ldb + j];
                    sum += (double)av * bv;
                }
                c[i * ldc + j] = accumulate ? c[i * ldc + j] + (float)sum : (float)sum;
            }
        }
    }
}
