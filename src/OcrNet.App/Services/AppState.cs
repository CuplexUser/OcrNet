using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using OcrNet.Core.Network;
using OcrNet.Core.Optimizers;
using OcrNet.Core.Training;
using OcrNet.Data.Datasets;
using OcrNet.Data.Glyphs;
using OcrNet.Data.Imaging;

namespace OcrNet.App.Services;

/// <summary>State shared by all tabs: the current model, fonts, dataset and training settings.</summary>
public sealed partial class AppState : ObservableObject
{
    public static readonly FontVariant[] AllVariants = [FontVariant.Regular, FontVariant.Bold, FontVariant.Italic, FontVariant.BoldItalic];

    private Task<List<FontFace>>? _fontsTask;
    private Task<GlyphCache>? _cacheTask;

    /// <summary>The network being designed, trained or used for recognition.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CharacterSet), nameof(Normalizer), nameof(ModelSummary), nameof(HasNetwork))]
    public partial NeuralNetwork? Network { get; set; }

    /// <summary>Optimizer state of the last training run (saved with the model to allow continuing).</summary>
    [ObservableProperty]
    public partial Optimizer? Optimizer { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(WindowTitle))]
    public partial string? ModelPath { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(WindowTitle))]
    public partial bool IsDirty { get; set; }

    [ObservableProperty]
    public partial bool IsTraining { get; set; }

    [ObservableProperty]
    public partial string Status { get; set; } = "Ready";

    /// <summary>Validation result of the most recent epoch.</summary>
    [ObservableProperty]
    public partial EvaluationResult? LastValidation { get; set; }

    /// <summary>The most recent normalized drawing from the Recognize tab (drives the visualizer).</summary>
    [ObservableProperty]
    public partial float[]? LastInput { get; set; }

    public ObservableCollection<EpochRecord> History { get; } = [];

    public TrainingOptions TrainingOptions { get; set; } = new();

    public DatasetOptions DatasetOptions { get; } = new();

    /// <summary>Selected style variants and font keys for training.</summary>
    public HashSet<FontVariant> SelectedVariants { get; } = [FontVariant.Regular, FontVariant.Bold, FontVariant.Italic];

    public HashSet<string>? SelectedFontKeys { get; set; }

    public UserSampleStore UserSamples { get; } = UserSampleStore.Load(UserSampleStore.DefaultPath);

    public bool HasNetwork => Network is not null;

    public CharacterSet CharacterSet => Network is null ? CharacterSet.DigitsOnly : CharacterSet.FromLabels(Network.Labels);

    /// <summary>Preprocessing used by the current model (stored in its metadata).</summary>
    public NormalizerOptions Normalizer => Network?.Metadata.Preprocessing is { Count: > 0 } p
        ? NormalizerOptions.FromDictionary(p)
        : DatasetOptions.Normalizer;

    public string ModelSummary => Network is null
        ? "No network"
        : $"{Network.Metadata.Name} · {Network.ClassCount} classes · {Network.ParameterCount:N0} parameters";

    public string WindowTitle =>
        $"OcrNet — {(ModelPath is null ? "untitled" : System.IO.Path.GetFileName(ModelPath))}{(IsDirty ? " •" : "")}";

    public event EventHandler? NetworkReplaced;

    /// <summary>Installs a new or loaded network and resets everything derived from the previous one.</summary>
    public void SetNetwork(NeuralNetwork network, Optimizer? optimizer, string? path)
    {
        Network = network;
        Optimizer = optimizer;
        ModelPath = path;
        IsDirty = path is null;
        LastValidation = null;
        History.Clear();
        foreach (var h in network.Metadata.History)
            History.Add(h);
        if (network.Metadata.Training.Count > 0)
            TrainingOptions = TrainingOptions.FromDictionary(network.Metadata.Training);
        NetworkReplaced?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>Raised after training changed the weights (visualizations should refresh).</summary>
    public event EventHandler? WeightsChanged;

    public void NotifyWeightsChanged() => WeightsChanged?.Invoke(this, EventArgs.Empty);

    /// <summary>Installed fonts (all style variants), discovered once in the background.</summary>
    public Task<List<FontFace>> GetFontsAsync() =>
        _fontsTask ??= Task.Run(() => FontCatalog.Discover(AllVariants));

    public Task<GlyphCache> GetGlyphCacheAsync() =>
        _cacheTask ??= Task.Run(() => GlyphCache.Load(GlyphCache.DefaultPath));

    /// <summary>Fonts currently selected for training.</summary>
    public async Task<List<FontFace>> GetSelectedFontsAsync()
    {
        var fonts = await GetFontsAsync();
        return [.. fonts.Where(f => SelectedVariants.Contains(f.Variant)
            && (SelectedFontKeys?.Contains(f.Family) ?? f.SelectedByDefault))];
    }

    /// <summary>
    /// Renders any missing glyphs for the selected fonts and builds the training data.
    /// Long-running; call from a background-friendly async context.
    /// </summary>
    public async Task<GlyphDataset> BuildDatasetAsync(IProgress<string>? status, CancellationToken cancellationToken)
    {
        var faces = await GetSelectedFontsAsync();
        if (faces.Count == 0)
            throw new InvalidOperationException("No fonts are selected on the Dataset tab.");
        var cache = await GetGlyphCacheAsync();
        var set = CharacterSet;
        var options = DatasetOptions.Clone();
        ulong seed = TrainingOptions.Seed;
        int threads = TrainingOptions.Threads;

        return await Task.Run(() =>
        {
            var renderProgress = new Progress<double>(f => status?.Report($"Rendering font glyphs… {f:P0}"));
            lock (cache)
            {
                if (cache.EnsureRendered(faces, set.Characters, threads, renderProgress, cancellationToken))
                {
                    status?.Report("Saving glyph cache…");
                    cache.Save(GlyphCache.DefaultPath);
                }
            }
            status?.Report("Preparing dataset…");
            return new GlyphDataset(set, faces, cache, options, UserSamples, seed);
        }, cancellationToken);
    }
}
