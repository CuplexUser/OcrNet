using OcrNet.Core.Compute;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Training;

/// <summary>Fused softmax + cross-entropy loss, the standard output for classification.</summary>
public static class SoftmaxCrossEntropy
{
    /// <summary>
    /// Computes the mean loss over the batch and writes dLoss/dLogits into <paramref name="gradient"/>
    /// (which may be null for evaluation). <paramref name="probabilities"/> receives the softmax output.
    /// </summary>
    public static float Compute(
        Tensor logits, ReadOnlySpan<int> labels, Tensor probabilities, Tensor? gradient,
        float labelSmoothing, out int correct)
    {
        int n = logits.Batch;
        int classes = logits.SampleSize;
        float offTarget = labelSmoothing / classes;
        float onTarget = 1f - labelSmoothing + offTarget;
        float invN = 1f / n;
        double loss = 0;
        correct = 0;

        for (int i = 0; i < n; i++)
        {
            var p = probabilities.Sample(i);
            logits.Sample(i).CopyTo(p);
            Ops.SoftmaxInPlace(p);

            int label = labels[i];
            int argMax = 0;
            for (int c = 1; c < classes; c++)
                if (p[c] > p[argMax])
                    argMax = c;
            if (argMax == label)
                correct++;

            if (labelSmoothing > 0f)
            {
                for (int c = 0; c < classes; c++)
                {
                    float t = c == label ? onTarget : offTarget;
                    loss -= t * MathF.Log(MathF.Max(p[c], 1e-12f));
                }
            }
            else
            {
                loss -= MathF.Log(MathF.Max(p[label], 1e-12f));
            }

            if (gradient is not null)
            {
                var g = gradient.Sample(i);
                for (int c = 0; c < classes; c++)
                {
                    float t = c == label ? onTarget : offTarget;
                    g[c] = (p[c] - t) * invN;
                }
            }
        }
        return (float)(loss / n);
    }
}
