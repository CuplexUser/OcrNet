using System.Runtime.CompilerServices;

namespace OcrNet.Core.Compute;

/// <summary>
/// xoshiro128** pseudo-random generator: small, fast and fully deterministic for a given seed,
/// which keeps training runs reproducible across machines and .NET versions.
/// </summary>
public sealed class FastRandom
{
    private uint _s0, _s1, _s2, _s3;
    private float _spareGaussian;
    private bool _hasSpare;

    public FastRandom(ulong seed)
    {
        // SplitMix64 expands the seed into the 128-bit state.
        ulong z = seed;
        _s0 = (uint)SplitMix(ref z);
        _s1 = (uint)SplitMix(ref z);
        _s2 = (uint)SplitMix(ref z);
        _s3 = (uint)SplitMix(ref z);
        if ((_s0 | _s1 | _s2 | _s3) == 0)
            _s0 = 1;
    }

    /// <summary>Derives an independent seed from a base seed and a list of stream identifiers.</summary>
    public static ulong Derive(ulong seed, params ReadOnlySpan<long> streams)
    {
        ulong z = seed;
        foreach (long s in streams)
        {
            z ^= (ulong)s + 0x9E3779B97F4A7C15UL + (z << 6) + (z >> 2);
            SplitMix(ref z);
        }
        return SplitMix(ref z);
    }

    private static ulong SplitMix(ref ulong x)
    {
        ulong z = x += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint NextUInt()
    {
        uint result = System.Numerics.BitOperations.RotateLeft(_s1 * 5, 7) * 9;
        uint t = _s1 << 9;
        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = System.Numerics.BitOperations.RotateLeft(_s3, 11);
        return result;
    }

    /// <summary>Uniform float in [0, 1).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float NextFloat() => (NextUInt() >> 8) * (1.0f / (1 << 24));

    /// <summary>Uniform float in [min, max).</summary>
    public float NextFloat(float min, float max) => min + (max - min) * NextFloat();

    /// <summary>Uniform integer in [0, maxExclusive).</summary>
    public int Next(int maxExclusive) => (int)(((ulong)NextUInt() * (ulong)maxExclusive) >> 32);

    /// <summary>Standard normal sample (Box–Muller).</summary>
    public float NextGaussian()
    {
        if (_hasSpare)
        {
            _hasSpare = false;
            return _spareGaussian;
        }
        float u, v, s;
        do
        {
            u = NextFloat() * 2f - 1f;
            v = NextFloat() * 2f - 1f;
            s = u * u + v * v;
        } while (s >= 1f || s == 0f);
        float m = MathF.Sqrt(-2f * MathF.Log(s) / s);
        _spareGaussian = v * m;
        _hasSpare = true;
        return u * m;
    }

    /// <summary>Fisher–Yates shuffle.</summary>
    public void Shuffle<T>(Span<T> items)
    {
        for (int i = items.Length - 1; i > 0; i--)
        {
            int j = Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }
}
