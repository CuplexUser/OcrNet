using OcrNet.Core.Network;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Training;

/// <summary>A validation sample the network got wrong.</summary>
public sealed record Misclassified(float[] Input, int Expected, int Predicted, float Confidence);

public sealed class EvaluationResult
{
    public required float Loss { get; init; }

    public required float Accuracy { get; init; }

    public required int Count { get; init; }

    /// <summary>[expected, predicted] counts.</summary>
    public required int[,] Confusion { get; init; }

    /// <summary>The most confidently wrong samples, worst first.</summary>
    public required IReadOnlyList<Misclassified> Worst { get; init; }
}

public static class Evaluator
{
    public static EvaluationResult Evaluate(NeuralNetwork network, IReadOnlyList<Batch> batches, int keepWorst = 48)
    {
        int classes = network.ClassCount;
        var confusion = new int[classes, classes];
        var worst = new List<Misclassified>();
        Tensor? probabilities = null;
        double lossSum = 0;
        int correctSum = 0, count = 0;

        foreach (var batch in batches)
        {
            lock (network.SyncRoot)
            {
                var logits = network.Forward(batch.Inputs, training: false);
                var probs = Tensor.Ensure(ref probabilities, batch.Count, logits.Shape);
                float loss = SoftmaxCrossEntropy.Compute(logits, batch.Labels.AsSpan(0, batch.Count), probs, null, 0f, out int correct);
                lossSum += loss * batch.Count;
                correctSum += correct;
                count += batch.Count;

                for (int i = 0; i < batch.Count; i++)
                {
                    var p = probs.Sample(i);
                    int predicted = 0;
                    for (int c = 1; c < classes; c++)
                        if (p[c] > p[predicted])
                            predicted = c;
                    int expected = batch.Labels[i];
                    confusion[expected, predicted]++;
                    if (predicted != expected && keepWorst > 0)
                    {
                        float confidence = p[predicted];
                        if (worst.Count < keepWorst || confidence > worst[^1].Confidence)
                        {
                            worst.Add(new Misclassified(batch.Inputs.Sample(i).ToArray(), expected, predicted, confidence));
                            worst.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
                            if (worst.Count > keepWorst)
                                worst.RemoveAt(worst.Count - 1);
                        }
                    }
                }
            }
        }

        return new EvaluationResult
        {
            Loss = count > 0 ? (float)(lossSum / count) : float.NaN,
            Accuracy = count > 0 ? (float)correctSum / count : float.NaN,
            Count = count,
            Confusion = confusion,
            Worst = worst,
        };
    }
}
