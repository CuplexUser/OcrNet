using OcrNet.Core.Compute;

namespace OcrNet.Core.Init;

public enum WeightInitKind
{
    /// <summary>He for ReLU-like activations, Xavier for tanh/sigmoid.</summary>
    Auto,
    He,
    Xavier,
    LeCun,
}

public static class WeightInit
{
    public static void Fill(Span<float> weights, int fanIn, int fanOut, WeightInitKind kind, FastRandom rng)
    {
        switch (kind)
        {
            case WeightInitKind.Xavier:
            {
                float limit = MathF.Sqrt(6f / (fanIn + fanOut));
                for (int i = 0; i < weights.Length; i++)
                    weights[i] = rng.NextFloat(-limit, limit);
                break;
            }
            case WeightInitKind.LeCun:
            {
                float std = MathF.Sqrt(1f / fanIn);
                for (int i = 0; i < weights.Length; i++)
                    weights[i] = rng.NextGaussian() * std;
                break;
            }
            default:
            {
                float std = MathF.Sqrt(2f / fanIn);
                for (int i = 0; i < weights.Length; i++)
                    weights[i] = rng.NextGaussian() * std;
                break;
            }
        }
    }
}
