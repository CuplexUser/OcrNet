using System.Numerics;
using OcrNet.Core.Compute;
using OcrNet.Core.Layers;

namespace OcrNet.Core.Optimizers;

/// <summary>RMSProp: divides the gradient by a running RMS of recent gradients.</summary>
public sealed class RmsPropOptimizer(float rho, float epsilon, float weightDecay) : Optimizer(weightDecay)
{
    public float Rho { get; } = rho;

    public float Epsilon { get; } = epsilon;

    public override OptimizerKind Kind => OptimizerKind.RmsProp;

    protected override int StateSlots => 1;

    protected override void Update(Parameter parameter, float[][] state, float learningRate)
    {
        float[] w = parameter.Value, g = parameter.Grad, s = state[0];
        float wd = parameter.ApplyWeightDecay ? WeightDecay : 0f;
        float rho = Rho, eps = Epsilon;

        Ops.ForChunks(w.Length, (start, count) =>
        {
            int i = start, end = start + count;
            int width = Vector<float>.Count;
            var vLr = new Vector<float>(learningRate);
            var vWd = new Vector<float>(wd);
            var vRho = new Vector<float>(rho);
            var vOneMinusRho = new Vector<float>(1f - rho);
            var vEps = new Vector<float>(eps);
            for (; i <= end - width; i += width)
            {
                var wi = new Vector<float>(w, i);
                var gi = new Vector<float>(g, i) + vWd * wi;
                var si = vRho * new Vector<float>(s, i) + vOneMinusRho * gi * gi;
                si.CopyTo(s, i);
                (wi - vLr * gi / (Vector.SquareRoot(si) + vEps)).CopyTo(w, i);
            }
            for (; i < end; i++)
            {
                float gi = g[i] + wd * w[i];
                s[i] = rho * s[i] + (1f - rho) * gi * gi;
                w[i] -= learningRate * gi / (MathF.Sqrt(s[i]) + eps);
            }
        });
    }
}
