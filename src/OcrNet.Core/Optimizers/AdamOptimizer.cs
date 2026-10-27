using System.Numerics;
using OcrNet.Core.Compute;
using OcrNet.Core.Layers;

namespace OcrNet.Core.Optimizers;

/// <summary>
/// Adam with bias correction. With <c>decoupled</c> set this is AdamW: weight decay shrinks the
/// weights directly instead of being added to the gradient as an L2 term.
/// </summary>
public sealed class AdamOptimizer(float beta1, float beta2, float epsilon, float weightDecay, bool decoupled)
    : Optimizer(weightDecay)
{
    public float Beta1 { get; } = beta1;

    public float Beta2 { get; } = beta2;

    public float Epsilon { get; } = epsilon;

    public bool Decoupled { get; } = decoupled;

    public override OptimizerKind Kind => Decoupled ? OptimizerKind.AdamW : OptimizerKind.Adam;

    protected override int StateSlots => 2;

    protected override void Update(Parameter parameter, float[][] state, float learningRate)
    {
        float[] w = parameter.Value, g = parameter.Grad, m = state[0], v = state[1];
        float wd = parameter.ApplyWeightDecay ? WeightDecay : 0f;
        float l2 = Decoupled ? 0f : wd;
        float decay = Decoupled ? learningRate * wd : 0f;
        float b1 = Beta1, b2 = Beta2, eps = Epsilon;
        float correction1 = 1f - MathF.Pow(b1, StepCount);
        float correction2 = 1f - MathF.Pow(b2, StepCount);
        // Fold the bias corrections into the step size: lr·m̂/(√v̂+ε) ≈ lr·√c2/c1 · m/(√v + ε·√c2)
        float stepSize = learningRate * MathF.Sqrt(correction2) / correction1;
        float epsHat = eps * MathF.Sqrt(correction2);

        Ops.ForChunks(w.Length, (start, count) =>
        {
            int i = start, end = start + count;
            int width = Vector<float>.Count;
            var vL2 = new Vector<float>(l2);
            var vDecay = new Vector<float>(decay);
            var vB1 = new Vector<float>(b1);
            var vB2 = new Vector<float>(b2);
            var v1B1 = new Vector<float>(1f - b1);
            var v1B2 = new Vector<float>(1f - b2);
            var vStep = new Vector<float>(stepSize);
            var vEps = new Vector<float>(epsHat);
            for (; i <= end - width; i += width)
            {
                var wi = new Vector<float>(w, i);
                var gi = new Vector<float>(g, i) + vL2 * wi;
                var mi = vB1 * new Vector<float>(m, i) + v1B1 * gi;
                var vi = vB2 * new Vector<float>(v, i) + v1B2 * gi * gi;
                mi.CopyTo(m, i);
                vi.CopyTo(v, i);
                (wi - vDecay * wi - vStep * mi / (Vector.SquareRoot(vi) + vEps)).CopyTo(w, i);
            }
            for (; i < end; i++)
            {
                float gi = g[i] + l2 * w[i];
                m[i] = b1 * m[i] + (1f - b1) * gi;
                v[i] = b2 * v[i] + (1f - b2) * gi * gi;
                w[i] -= decay * w[i] + stepSize * m[i] / (MathF.Sqrt(v[i]) + epsHat);
            }
        });
    }
}
