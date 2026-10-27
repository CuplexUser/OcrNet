using System.Numerics;
using OcrNet.Core.Compute;
using OcrNet.Core.Layers;

namespace OcrNet.Core.Optimizers;

/// <summary>Stochastic gradient descent with optional (Nesterov) momentum and L2 weight decay.</summary>
public sealed class SgdOptimizer(float momentum, bool nesterov, float weightDecay) : Optimizer(weightDecay)
{
    public float Momentum { get; } = momentum;

    public bool Nesterov { get; } = nesterov;

    public override OptimizerKind Kind => OptimizerKind.Sgd;

    protected override int StateSlots => 1;

    protected override void Update(Parameter parameter, float[][] state, float learningRate)
    {
        float[] w = parameter.Value, g = parameter.Grad, v = state[0];
        float wd = parameter.ApplyWeightDecay ? WeightDecay : 0f;
        float mu = Momentum;
        bool nesterov = Nesterov;

        Ops.ForChunks(w.Length, (start, count) =>
        {
            int i = start, end = start + count;
            int width = Vector<float>.Count;
            var vLr = new Vector<float>(learningRate);
            var vWd = new Vector<float>(wd);
            var vMu = new Vector<float>(mu);
            for (; i <= end - width; i += width)
            {
                var wi = new Vector<float>(w, i);
                var gi = new Vector<float>(g, i) + vWd * wi;
                var vi = vMu * new Vector<float>(v, i) + gi;
                vi.CopyTo(v, i);
                var step = nesterov ? gi + vMu * vi : vi;
                (wi - vLr * step).CopyTo(w, i);
            }
            for (; i < end; i++)
            {
                float gi = g[i] + wd * w[i];
                v[i] = mu * v[i] + gi;
                w[i] -= learningRate * (nesterov ? gi + mu * v[i] : v[i]);
            }
        });
    }
}
