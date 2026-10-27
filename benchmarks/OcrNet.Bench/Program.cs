using System.Diagnostics;
using System.IO;
using OcrNet.Core.Compute;
using OcrNet.Core.Network;
using OcrNet.Core.Tensors;
using OcrNet.Core.Training;
using OcrNet.Data.Datasets;
using OcrNet.Data.Glyphs;

// Usage: OcrNet.Bench [gemm] [mlp] [digits] [balanced] [alnum]   (default: gemm mlp)
var modes = args.Length > 0 ? args : ["gemm", "mlp"];
Console.WriteLine($"CPU threads: {Environment.ProcessorCount}, AVX2: {System.Runtime.Intrinsics.X86.Avx2.IsSupported}, FMA: {System.Runtime.Intrinsics.X86.Fma.IsSupported}");

if (modes.Contains("gemm"))
    BenchGemm();
if (modes.Contains("mlp"))
    BenchMlp();
if (modes.Contains("pipeline"))
    BenchPipeline();
if (modes.Contains("fonts"))
    foreach (var f in FontCatalog.Discover([FontVariant.Regular]))
        Console.WriteLine($"  {f.DisplayName}{(f.UseLowercase ? "" : "  [no lowercase]")}");
if (modes.Contains("digits"))
    BenchFonts(CharacterSet.DigitsOnly, [256, 128], epochs: 8, samplesPerClass: 2000);
if (modes.Contains("balanced"))
    BenchFonts(CharacterSet.Balanced, [512, 256], epochs: 15, samplesPerClass: 1200);
if (modes.Contains("alnum"))
    BenchFonts(CharacterSet.Alphanumeric, [512, 256], epochs: 15, samplesPerClass: 1200);

static void BenchPipeline()
{
    Console.WriteLine("\nData pipeline throughput (augment + normalize only)");
    var faces = FontCatalog.Discover([FontVariant.Regular]);
    var cache = GlyphCache.Load(GlyphCache.DefaultPath);
    cache.EnsureRendered(faces, CharacterSet.Alphanumeric.Characters, Environment.ProcessorCount);
    var data = new GlyphDataset(CharacterSet.Alphanumeric, faces, cache, new DatasetOptions { SamplesPerClass = 500 }, null, 1);
    foreach (int threads in new[] { 1, 4, 8 })
    {
        Parallelism.MaxThreads = threads;
        var sw = Stopwatch.StartNew();
        int n = 0;
        foreach (var b in data.GetTrainBatches(1, 128, 1, CancellationToken.None))
            n += b.Count;
        Console.WriteLine($"  MaxThreads {threads}: {n / sw.Elapsed.TotalSeconds:N0} samples/s");
    }
    Parallelism.MaxThreads = Environment.ProcessorCount;

    // Per-stage cost on one thread.
    var glyph = faces.Select(f => cache.TryGet(f, 'R', out var g) ? g : null).First(g => g is not null)!;
    var rng = new FastRandom(9);
    var augmentation = new OcrNet.Data.Imaging.AugmentationOptions();
    var scratch = new OcrNet.Data.Imaging.GrayImage(64, 64);
    var output = new float[784];
    var normalizer = new OcrNet.Data.Imaging.NormalizerOptions();
    const int reps = 5000;
    var t = Stopwatch.StartNew();
    for (int i = 0; i < reps; i++) OcrNet.Data.Imaging.Augmenter.Distort(glyph, augmentation, rng, scratch);
    Console.WriteLine($"  distort:   {t.Elapsed.TotalMicroseconds / reps,6:F1} µs");
    foreach (var (label, variant) in new[]
    {
        ("affine only", new OcrNet.Data.Imaging.AugmentationOptions { ElasticProbability = 0, StrokeProbability = 0 }),
        ("+ elastic  ", new OcrNet.Data.Imaging.AugmentationOptions { ElasticProbability = 1, StrokeProbability = 0 }),
        ("+ stroke   ", new OcrNet.Data.Imaging.AugmentationOptions { ElasticProbability = 0, StrokeProbability = 1 }),
    })
    {
        t.Restart();
        for (int i = 0; i < reps; i++) OcrNet.Data.Imaging.Augmenter.Distort(glyph, variant, rng, scratch);
        Console.WriteLine($"    {label}: {t.Elapsed.TotalMicroseconds / reps,6:F1} µs");
    }
    t.Restart();
    for (int i = 0; i < reps; i++) OcrNet.Data.Imaging.Normalizer.Normalize(scratch, normalizer, output);
    Console.WriteLine($"  normalize: {t.Elapsed.TotalMicroseconds / reps,6:F1} µs");
    t.Restart();
    for (int i = 0; i < reps; i++) OcrNet.Data.Imaging.Augmenter.AddNoise(output, 0.02f, rng);
    Console.WriteLine($"  noise:     {t.Elapsed.TotalMicroseconds / reps,6:F1} µs");
}

static void BenchFonts(CharacterSet set, int[] hidden, int epochs, int samplesPerClass)
{
    Console.WriteLine($"\nEnd-to-end font training: {set}");
    var sw = Stopwatch.StartNew();
    var faces = FontCatalog.Discover([FontVariant.Regular, FontVariant.Bold, FontVariant.Italic]).Where(f => f.SelectedByDefault).ToList();
    Console.WriteLine($"  {faces.Count} font faces ({faces.Select(f => f.Family).Distinct().Count()} families), {faces.Count(f => !f.UseLowercase)} without usable lowercase, discovered in {sw.Elapsed.TotalSeconds:F1}s");
    sw.Restart();
    var cache = GlyphCache.Load(GlyphCache.DefaultPath);
    if (cache.EnsureRendered(faces, set.Characters, Environment.ProcessorCount))
        cache.Save(GlyphCache.DefaultPath);
    Console.WriteLine($"  {cache.Count} glyphs ready in {sw.Elapsed.TotalSeconds:F1}s");

    var data = new GlyphDataset(set, faces, cache, new DatasetOptions { SamplesPerClass = samplesPerClass }, null, 42);
    Console.WriteLine($"  train: {data.TrainFontCount} faces / {data.TrainGlyphCount} glyphs, validation: {data.ValidationFontCount} faces / {data.ValidationSampleCount} samples");
    var network = NetworkBuilder.Build(new NetworkSpec
    {
        Labels = [.. set.Classes],
        HiddenLayers = [.. hidden.Select(u => new HiddenLayerSpec { Units = u, Dropout = 0.2f })],
    });
    var options = new TrainingOptions { Epochs = epochs, EarlyStoppingPatience = 0 };
    var progress = new ConsoleProgress();
    var result = new Trainer(network, data, options, progress).Run(CancellationToken.None);
    Console.WriteLine($"  done in {result.Duration.TotalSeconds:F0}s, best epoch {result.BestEpoch}, validation accuracy {result.FinalValidation?.Accuracy:P2}");
    if (result.FinalValidation is { } v)
    {
        var confusions = new List<(int Count, string Pair)>();
        for (int a = 0; a < set.Count; a++)
            for (int b = 0; b < set.Count; b++)
                if (a != b && v.Confusion[a, b] > 0)
                    confusions.Add((v.Confusion[a, b], $"{CharacterSet.Display(set.Classes[a])}→{CharacterSet.Display(set.Classes[b])}"));
        Console.WriteLine("  top confusions: " + string.Join(", ", confusions.OrderByDescending(c => c.Count).Take(10).Select(c => $"{c.Pair} ({c.Count})")));
    }
    OcrNet.Core.Serialization.ModelFile.Save(Path.Combine(Path.GetTempPath(), $"bench-{set.Count}.ocrnet"), network);
}

static void BenchGemm()
{
    Console.WriteLine("\nSGEMM (shapes of a 784-512-256-62 MLP, batch 128)");
    (string name, bool ta, bool tb, int m, int n, int k)[] cases =
    [
        ("fwd  X·Wᵀ   128×512×784", false, true, 128, 512, 784),
        ("bwd  dYᵀ·X  512×784×128", true, false, 512, 784, 128),
        ("bwd  dY·W   128×256×512", false, false, 128, 256, 512),
        ("square      1024³      ", false, false, 1024, 1024, 1024),
    ];
    var rng = new FastRandom(1);
    foreach (var (name, ta, tb, m, n, k) in cases)
    {
        var a = Fill(rng, m * k);
        var b = Fill(rng, k * n);
        var c = new float[m * n];
        int lda = ta ? m : k, ldb = tb ? k : n;
        for (int w = 0; w < 3; w++)
            Gemm.Multiply(ta, tb, m, n, k, a, lda, b, ldb, c, n);
        double flops = 2.0 * m * n * k;
        int reps = (int)Math.Max(5, 4e9 / flops);
        var sw = Stopwatch.StartNew();
        for (int r = 0; r < reps; r++)
            Gemm.Multiply(ta, tb, m, n, k, a, lda, b, ldb, c, n);
        sw.Stop();
        Console.WriteLine($"  {name}: {flops * reps / sw.Elapsed.TotalSeconds / 1e9,6:F1} GFLOPS");
    }
}

static void BenchMlp()
{
    Console.WriteLine("\nMLP training throughput (random data, no augmentation cost)");
    foreach (var (name, hidden, classes) in new[]
    {
        ("784-256-128-10    ", new[] { 256, 128 }, 10),
        ("784-512-256-62    ", new[] { 512, 256 }, 62),
        ("784-1024-512-256-62", new[] { 1024, 512, 256 }, 62),
    })
    {
        var labels = Enumerable.Range(0, classes).Select(i => i.ToString()).ToList();
        var network = NetworkBuilder.Build(new NetworkSpec
        {
            Labels = labels,
            HiddenLayers = [.. hidden.Select(u => new HiddenLayerSpec { Units = u, Dropout = 0.2f })],
        });
        int samples = 20_000;
        var rng = new FastRandom(2);
        var x = new float[samples][];
        var y = new int[samples];
        for (int i = 0; i < samples; i++)
        {
            x[i] = Fill(rng, 784);
            y[i] = rng.Next(classes);
        }
        var data = new InMemoryTrainingData(new TensorShape(1, 28, 28), classes, x, y, [], []);
        var options = new TrainingOptions { Epochs = 3, BatchSize = 128, EarlyStoppingPatience = 0 };
        var sw = Stopwatch.StartNew();
        new Trainer(network, data, options).Run(CancellationToken.None);
        sw.Stop();
        double perSecond = samples * options.Epochs / sw.Elapsed.TotalSeconds;
        double gflops = perSecond * network.FlopsPerSample * 3 / 1e9;
        Console.WriteLine($"  {name}: {perSecond,9:N0} samples/s  (~{gflops:F1} GFLOPS effective, {network.ParameterCount:N0} params)");
    }
}

static float[] Fill(FastRandom rng, int length)
{
    var data = new float[length];
    for (int i = 0; i < length; i++)
        data[i] = rng.NextFloat(-1f, 1f);
    return data;
}

sealed class ConsoleProgress : IProgress<TrainingProgress>
{
    public void Report(TrainingProgress p)
    {
        if (p.CompletedEpoch is { } e)
            Console.WriteLine($"  epoch {e.Epoch,2}: loss {e.TrainLoss:F3} acc {e.TrainAccuracy:P1} | val loss {e.ValidationLoss:F3} acc {e.ValidationAccuracy:P2} | lr {e.LearningRate:G3} | {e.Seconds:F1}s ({p.SamplesPerSecond:N0}/s)");
    }
}
