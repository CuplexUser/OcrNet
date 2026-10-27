using OcrNet.Core.Compute;
using OcrNet.Core.Tensors;
using OcrNet.Core.Training;
using OcrNet.Data.Glyphs;
using OcrNet.Data.Imaging;

namespace OcrNet.Data.Datasets;

public sealed class DatasetOptions
{
    /// <summary>Training samples generated per class per epoch (each a freshly augmented glyph).</summary>
    public int SamplesPerClass { get; set; } = 1200;

    /// <summary>Fraction of font families held out for validation, so accuracy is measured on unseen fonts.</summary>
    public float ValidationFontFraction { get; set; } = 0.15f;

    /// <summary>Augmented copies per validation glyph in addition to the clean one.</summary>
    public int ValidationAugmentations { get; set; } = 2;

    /// <summary>Probability that a training sample of a class uses one of the user's drawings, when available.</summary>
    public float UserSampleProbability { get; set; } = 0.3f;

    public AugmentationOptions Augmentation { get; set; } = new();

    public NormalizerOptions Normalizer { get; set; } = new();

    public DatasetOptions Clone()
    {
        var copy = (DatasetOptions)MemberwiseClone();
        copy.Augmentation = Augmentation.Clone();
        copy.Normalizer = NormalizerOptions.FromDictionary(Normalizer.ToDictionary());
        return copy;
    }
}

/// <summary>
/// Training data generated from rendered font glyphs (plus optional user drawings). Each epoch draws
/// <see cref="DatasetOptions.SamplesPerClass"/> balanced samples per class, applies random
/// augmentation and the shared <see cref="Normalizer"/>. Validation uses held-out font families.
/// </summary>
public sealed class GlyphDataset : ITrainingData
{
    private readonly CharacterSet _set;
    private readonly DatasetOptions _options;
    private readonly GrayImage[][] _trainSources;   // per class
    private readonly GrayImage[][] _userSources;    // per class
    private readonly List<(GrayImage Image, int Label)> _validationSources = [];
    private List<Batch>? _validationBatches;
    private int _validationBatchSize;

    public GlyphDataset(CharacterSet set, IReadOnlyList<FontFace> faces, GlyphCache cache, DatasetOptions options,
        UserSampleStore? userSamples, ulong seed)
    {
        _set = set;
        _options = options.Clone();
        int size = _options.Normalizer.OutputSize;
        InputShape = new TensorShape(1, size, size);

        // Split by family so bold/italic variants of a validation font never leak into training.
        var families = faces.Select(f => f.Family).Distinct().OrderBy(f => f, StringComparer.Ordinal).ToArray();
        new FastRandom(FastRandom.Derive(seed, 1)).Shuffle(families.AsSpan());
        int validationCount = families.Length >= 5
            ? Math.Max(1, (int)Math.Round(families.Length * _options.ValidationFontFraction))
            : 0;
        var validationFamilies = families.Take(validationCount).ToHashSet();
        ValidationFamilies = [.. validationFamilies.Order()];

        var train = Enumerable.Range(0, set.Count).Select(_ => new List<GrayImage>()).ToArray();
        foreach (var face in faces)
        {
            bool validation = validationFamilies.Contains(face.Family);
            foreach (char c in set.Characters)
            {
                if (char.IsLower(c) && !face.UseLowercase)
                    continue;
                if (!cache.TryGet(face, c, out var image))
                    continue;
                int label = set.ClassOf(c);
                if (validation)
                    _validationSources.Add((image, label));
                else
                    train[label].Add(image);
            }
        }

        var missing = Enumerable.Range(0, set.Count).Where(i => train[i].Count == 0).Select(i => set.Classes[i]).ToList();
        if (missing.Count > 0)
            throw new InvalidOperationException(
                $"No selected training font provides: {string.Join(", ", missing.Select(CharacterSet.Display))}. Select more fonts.");
        _trainSources = [.. train.Select(l => l.ToArray())];

        var user = Enumerable.Range(0, set.Count).Select(_ => new List<GrayImage>()).ToArray();
        foreach (var sample in userSamples?.Samples ?? [])
        {
            int label = set.ClassOf(sample.Character);
            if (label >= 0)
                user[label].Add(sample.Image);
        }
        _userSources = [.. user.Select(l => l.ToArray())];

        TrainFontCount = faces.Count(f => !validationFamilies.Contains(f.Family));
        ValidationFontCount = faces.Count - TrainFontCount;
        TrainGlyphCount = _trainSources.Sum(s => s.Length);
        UserSampleCount = _userSources.Sum(s => s.Length);
    }

    public TensorShape InputShape { get; }

    public int ClassCount => _set.Count;

    public int TrainSamplesPerEpoch => _options.SamplesPerClass * _set.Count;

    public IReadOnlyList<string> ValidationFamilies { get; }

    public int TrainFontCount { get; }

    public int ValidationFontCount { get; }

    public int TrainGlyphCount { get; }

    public int UserSampleCount { get; }

    public int ValidationSampleCount => _validationSources.Count * (1 + _options.ValidationAugmentations);

    public IEnumerable<Batch> GetTrainBatches(int epoch, int batchSize, ulong seed, CancellationToken cancellationToken)
    {
        int total = TrainSamplesPerEpoch;
        var labels = new int[total];
        for (int i = 0; i < total; i++)
            labels[i] = i % _set.Count;
        new FastRandom(FastRandom.Derive(seed, 2, epoch)).Shuffle(labels.AsSpan());

        int batchCount = (total + batchSize - 1) / batchSize;
        // Roughly one producer per three compute threads keeps the GEMMs fed on this workload.
        int producers = Math.Max(1, (Parallelism.MaxThreads + 1) / 3);
        return BatchPipeline.Run(batchCount, batchSize, InputShape, producers, (index, batch) =>
        {
            var rng = new FastRandom(FastRandom.Derive(seed, 3, epoch, index));
            int start = index * batchSize;
            int count = Math.Min(batchSize, total - start);
            batch.Resize(count);
            for (int i = 0; i < count; i++)
            {
                int label = labels[start + i];
                var user = _userSources[label];
                GrayImage source = user.Length > 0 && rng.NextFloat() < _options.UserSampleProbability
                    ? user[rng.Next(user.Length)]
                    : _trainSources[label][rng.Next(_trainSources[label].Length)];
                MakeSample(source, rng, augment: true, batch.Inputs.Sample(i));
                batch.Labels[i] = label;
            }
        }, cancellationToken);
    }

    public IReadOnlyList<Batch> GetValidationBatches(int batchSize)
    {
        if (_validationBatches is not null && _validationBatchSize == batchSize)
            return _validationBatches;

        int copies = 1 + _options.ValidationAugmentations;
        int total = _validationSources.Count * copies;
        var inputs = new float[total][];
        var labels = new int[total];
        Parallel.For(0, _validationSources.Count, Parallelism.Options, i =>
        {
            var rng = new FastRandom(FastRandom.Derive(4, i));
            var (image, label) = _validationSources[i];
            for (int k = 0; k < copies; k++)
            {
                var input = new float[InputShape.Size];
                MakeSample(image, rng, augment: k > 0, input);
                inputs[i * copies + k] = input;
                labels[i * copies + k] = label;
            }
        });
        _validationBatches = InMemoryTrainingData.ToBatches(inputs, labels, InputShape, batchSize);
        _validationBatchSize = batchSize;
        return _validationBatches;
    }

    /// <summary>Augmented, normalized samples for the dataset preview.</summary>
    public List<(float[] Input, int Label)> Preview(int count, ulong seed)
    {
        var rng = new FastRandom(seed);
        var result = new List<(float[], int)>();
        for (int i = 0; i < count; i++)
        {
            int label = rng.Next(_set.Count);
            var sources = _trainSources[label];
            var input = new float[InputShape.Size];
            MakeSample(sources[rng.Next(sources.Length)], rng, augment: true, input);
            result.Add((input, label));
        }
        return result;
    }

    [ThreadStatic]
    private static GrayImage? _scratch;

    private void MakeSample(GrayImage source, FastRandom rng, bool augment, Span<float> output)
    {
        var image = source;
        if (augment)
        {
            if (_scratch is null || _scratch.Width != source.Width || _scratch.Height != source.Height)
                _scratch = new GrayImage(source.Width, source.Height);
            Augmenter.Distort(source, _options.Augmentation, rng, _scratch);
            image = _scratch;
        }
        Normalizer.Normalize(image, _options.Normalizer, output);
        if (augment && _options.Augmentation.Enabled)
            Augmenter.AddNoise(output, _options.Augmentation.Noise, rng);
    }
}
