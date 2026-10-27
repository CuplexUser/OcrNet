using OcrNet.Core.Init;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Data.Datasets;

namespace OcrNet.App.Services;

public enum AdviceLevel
{
    Good,
    Tip,
    Warning,
}

public sealed record DesignAdvice(AdviceLevel Level, string Text);

/// <summary>
/// Rule-of-thumb guidance for a hidden-layer design given the output classes. The figures quoted
/// come from the measurements in README.md (held-out fonts, 8 threads on a 4-core laptop).
/// </summary>
public static class DesignAdvisor
{
    /// <summary>Characters that look alike as isolated glyphs; ambiguous whenever they are separate classes.</summary>
    private static readonly (char A, char B)[] LookAlikes =
    [
        ('0', 'O'), ('1', 'I'), ('1', 'l'), ('I', 'l'), ('2', 'Z'), ('5', 'S'), ('8', 'B'), ('/', '1'), ('(', 'C'),
        .. (CharacterSet.MergedLower + CharacterSet.SwedishLower).Select(c => (char.ToUpperInvariant(c), c)),
    ];

    /// <summary>A starting-point architecture for this many classes, with the reasoning.</summary>
    public static string Recommend(CharacterSet set)
    {
        int n = set.Count;
        if (n <= 12)
            return $"{n} classes is an easy problem: one or two hidden layers of 128–256 units are enough. " +
                   "“Digits – fast” (256 → 128) reaches about 98 % on unseen fonts in under 10 seconds; " +
                   "wider or deeper networks mostly add training time.";
        if (n <= 36)
            return $"{n} classes: use two hidden layers of about 384–512 then 192–256 units. Width matters more " +
                   "than depth, because the last hidden layer must offer distinct features for every class.";
        string text = $"{n} classes needs more capacity: two hidden layers of about 512 → 256, or 1024 → 512 → 256 " +
                      "with batch normalization. “Balanced 47” (512 → 256) reaches about 93 % on unseen fonts in under a minute.";
        if (n > 62)
            text += $" With more than 62 classes, keep the last hidden layer at least {4 * n} units wide.";
        return text;
    }

    public static IReadOnlyList<DesignAdvice> Review(CharacterSet set, IReadOnlyList<HiddenLayerSpec> layers, WeightInitKind init, long parameters)
    {
        int n = set.Count;
        var advice = new List<DesignAdvice>();
        void Warn(string text) => advice.Add(new(AdviceLevel.Warning, text));
        void Tip(string text) => advice.Add(new(AdviceLevel.Tip, text));

        ReviewClasses(set, Warn, Tip);

        if (layers.Count == 0)
        {
            Warn("No hidden layers: the network is a linear classifier that can learn only one template per class. " +
                 "It trains in seconds but cannot model the many ways each character is written. Add at least one hidden layer.");
            return Finish(advice, n);
        }

        var (minFirst, maxFirst) = n <= 12 ? (128, 512) : n <= 36 ? (256, 1024) : (384, 1536);
        int first = layers[0].Units;
        if (first < minFirst)
            Warn($"The first hidden layer ({first} units) is small for {n} classes. It has to detect every stroke and curve " +
                 $"the characters are made of; with too few units both training and validation accuracy stay low. Try at least {minFirst}.");
        else if (first > 2 * maxFirst)
            Tip($"A {first}-unit first layer is very wide. Training time grows with width, while accuracy gains above about {maxFirst} units are small.");

        int narrowest = Enumerable.Range(0, layers.Count).MinBy(i => layers[i].Units);
        int floor = Math.Max(2 * n, 32);
        if (layers[narrowest].Units < floor)
            Warn($"Layer {narrowest + 1} has only {layers[narrowest].Units} units for {n} classes. It is a bottleneck: the output layer " +
                 $"must separate {n} classes using just those features. Keep every hidden layer at {floor} units or more, ideally {4 * n} or more.");

        for (int i = 1; i < layers.Count; i++)
            if (layers[i].Units > layers[i - 1].Units * 1.5)
            {
                Tip($"Layer {i + 1} is wider than the layer before it. A funnel (each layer the same width or narrower) usually works " +
                    "better for classification: a later layer cannot recover detail an earlier one discarded.");
                break;
            }

        if (layers.Count >= 4)
            Tip($"{layers.Count} hidden layers: a fully connected network on 28×28 glyphs rarely gains from more than 2–3 layers, " +
                "and each extra layer adds training time.");
        if (layers.Count >= 3 && layers.Any(l => !l.BatchNorm))
            Warn("Without batch normalization on every layer, stacks of three or more layers train more slowly and are sensitive " +
                 "to the learning rate. Turn on BN for each layer.");

        if (layers.Any(l => l.Activation == ActivationKind.Sigmoid))
            Warn("Sigmoid saturates, so its gradients vanish and it learns much more slowly than ReLU in hidden layers. Use ReLU, LeakyReLU or GELU.");
        else if (layers.Any(l => l.Activation == ActivationKind.Tanh) && layers.Count >= 3)
            Tip("Tanh saturates and slows learning in deeper stacks. ReLU or GELU usually trains faster to the same accuracy.");
        if (layers.Any(l => l.Activation == ActivationKind.Gelu))
            Tip("GELU is smooth and sometimes slightly more accurate than ReLU, but it costs more time per epoch.");

        if (layers.FirstOrDefault(l => l.Dropout > 0.5f) is { } heavy)
            Warn($"Dropout {heavy.Dropout:0.##} on layer {layers.ToList().IndexOf(heavy) + 1} discards most of the layer's features on " +
                 "every step. The network underfits and needs many more epochs. Values of 0.1–0.3 work well.");
        else if (layers.All(l => l.Dropout == 0))
            Tip("No dropout. Each epoch draws newly augmented samples, so overfitting is mild, but a dropout of 0.1–0.3 usually " +
                "raises accuracy on unseen fonts and handwriting because it stops the network from relying on single features.");

        bool saturating = layers.Any(l => l.Activation is ActivationKind.Tanh or ActivationKind.Sigmoid);
        if (init == WeightInitKind.He && saturating)
            Tip("He initialization is tuned for ReLU. With Tanh or Sigmoid, choose Auto or Xavier.");
        else if (init is WeightInitKind.Xavier or WeightInitKind.LeCun && !saturating && layers.Count >= 2)
            Tip($"{init} initialization with ReLU-family activations starts with signals that shrink layer by layer. Auto picks He for these layers.");

        if (parameters > 3_000_000)
            Tip($"{parameters / 1e6:0.#} M parameters. Larger models need more epochs and samples per class to pay off; " +
                "on this data the gains above 1–2 M parameters are small.");

        return Finish(advice, n);
    }

    private static void ReviewClasses(CharacterSet set, Action<string> warn, Action<string> tip)
    {
        var pairs = LookAlikes
            .Where(p => set.ClassOf(p.A) is var a and >= 0 && set.ClassOf(p.B) is var b and >= 0 && a != b)
            .Select(p => $"{p.A}/{p.B}")
            .ToList();
        if (pairs.Count == 0)
            return;
        static bool IsCasePair(string p) => char.ToUpperInvariant(p[0]) == char.ToUpperInvariant(p[2]);
        int casePairs = pairs.Count(IsCasePair);
        if (casePairs >= 5)
            pairs = [.. pairs.OrderByDescending(IsCasePair)]; // the case pairs are what the warning is about
        string list = string.Join(", ", pairs.Take(8)) + (pairs.Count > 8 ? $" and {pairs.Count - 8} more" : "");
        if (casePairs >= 5)
            warn($"This set keeps upper and lowercase look-alikes apart ({list}). As isolated glyphs they are often identical, " +
                 "so accuracy levels off around 85 % (measured with 62 classes) however large the network. The limit is in the " +
                 "data, not the model. The Balanced sets merge these pairs.");
        else
            tip($"Look-alikes in this set: {list}. Isolated, they can be truly ambiguous, so they will dominate the confusion matrix " +
                "and cap accuracy a little no matter how the network is designed.");
    }

    private static IReadOnlyList<DesignAdvice> Finish(List<DesignAdvice> advice, int classes)
    {
        if (advice.All(a => a.Level != AdviceLevel.Warning))
            advice.Insert(0, new(AdviceLevel.Good, $"A sensible design for {classes} classes."));
        return advice;
    }
}
