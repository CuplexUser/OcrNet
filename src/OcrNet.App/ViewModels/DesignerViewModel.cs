using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OcrNet.App.Services;
using OcrNet.Core.Init;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Data.Datasets;

namespace OcrNet.App.ViewModels;

/// <summary>Editable row for one hidden layer.</summary>
public sealed partial class HiddenLayerItem : ObservableObject
{
    [ObservableProperty]
    public partial int Units { get; set; } = 256;

    [ObservableProperty]
    public partial ActivationKind Activation { get; set; } = ActivationKind.ReLU;

    [ObservableProperty]
    public partial float Dropout { get; set; } = 0.2f;

    [ObservableProperty]
    public partial bool BatchNorm { get; set; }

    public HiddenLayerSpec ToSpec() => new()
    {
        Units = Math.Clamp(Units, 1, 8192),
        Activation = Activation,
        Dropout = Math.Clamp(Dropout, 0f, 0.9f),
        BatchNorm = BatchNorm,
    };

    public static HiddenLayerItem From(HiddenLayerSpec spec) => new()
    {
        Units = spec.Units, Activation = spec.Activation, Dropout = spec.Dropout, BatchNorm = spec.BatchNorm,
    };
}

/// <summary>Network designer: character set, depth/width of hidden layers, activations and initialization.</summary>
public sealed partial class DesignerViewModel : ObservableObject
{
    private const string CustomSetName = "Custom…";
    private readonly AppState _state;
    private bool _loading;

    public DesignerViewModel(AppState state)
    {
        _state = state;
        Layers.CollectionChanged += (_, e) =>
        {
            foreach (HiddenLayerItem item in e.NewItems ?? Array.Empty<HiddenLayerItem>())
                item.PropertyChanged += OnLayerChanged;
            UpdatePreview();
        };
        state.NetworkReplaced += (_, _) => LoadFromNetwork();
        SelectedPreset = Presets[0];
    }

    public IReadOnlyList<Preset> Presets => Preset.All;

    public IReadOnlyList<string> CharacterSetNames { get; } = [.. CharacterSet.Presets.Select(p => p.Name), CustomSetName];

    public IReadOnlyList<ActivationKind> Activations { get; } = Enum.GetValues<ActivationKind>();

    public IReadOnlyList<WeightInitKind> WeightInits { get; } = Enum.GetValues<WeightInitKind>();

    public ObservableCollection<HiddenLayerItem> Layers { get; } = [];

    [ObservableProperty]
    public partial Preset? SelectedPreset { get; set; }

    [ObservableProperty]
    public partial string ModelName { get; set; } = "Untitled";

    [ObservableProperty]
    public partial string Description { get; set; } = "";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsCustomSet))]
    public partial string SelectedCharacterSetName { get; set; } = CharacterSet.DigitsOnly.Name;

    [ObservableProperty]
    public partial string CustomCharacters { get; set; } = "0123456789+-=";

    [ObservableProperty]
    public partial WeightInitKind WeightInit { get; set; } = WeightInitKind.Auto;

    [ObservableProperty]
    public partial ulong Seed { get; set; } = 42;

    /// <summary>Untrained network built from the current settings, for the summary and diagram.</summary>
    [ObservableProperty]
    public partial NeuralNetwork? Preview { get; set; }

    [ObservableProperty]
    public partial string Summary { get; set; } = "";

    [ObservableProperty]
    public partial string? Error { get; set; }

    public bool IsCustomSet => SelectedCharacterSetName == CustomSetName;

    public CharacterSet CurrentCharacterSet => IsCustomSet
        ? CharacterSet.Custom(CustomCharacters)
        : CharacterSet.Presets.First(p => p.Name == SelectedCharacterSetName);

    partial void OnSelectedPresetChanged(Preset? value)
    {
        if (value is not null)
            ApplyPreset(value);
    }

    partial void OnSelectedCharacterSetNameChanged(string value) => UpdatePreview();

    partial void OnCustomCharactersChanged(string value) => UpdatePreview();

    partial void OnWeightInitChanged(WeightInitKind value) => UpdatePreview();

    partial void OnSeedChanged(ulong value) => UpdatePreview();

    private void OnLayerChanged(object? sender, PropertyChangedEventArgs e) => UpdatePreview();

    private void ApplyPreset(Preset preset)
    {
        _loading = true;
        SelectedCharacterSetName = preset.CharacterSet.Name;
        ModelName = preset.Name.Split(' ')[0] + " model";
        Description = preset.Description;
        Layers.Clear();
        foreach (var layer in preset.HiddenLayers)
            Layers.Add(HiddenLayerItem.From(layer));
        _state.TrainingOptions = preset.ApplyTo(_state.TrainingOptions);
        _state.DatasetOptions.SamplesPerClass = preset.SamplesPerClass;
        _loading = false;
        UpdatePreview();
        TrainingOptionsReplaced?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>Raised when a preset changed the training options (the training tab rebinds).</summary>
    public event EventHandler? TrainingOptionsReplaced;

    private NetworkSpec CreateSpec() => new()
    {
        Labels = [.. CurrentCharacterSet.Classes],
        HiddenLayers = [.. Layers.Select(l => l.ToSpec())],
        WeightInit = WeightInit,
        Seed = Seed,
    };

    private void UpdatePreview()
    {
        if (_loading)
            return;
        try
        {
            var network = NetworkBuilder.Build(CreateSpec());
            Preview = network;
            Error = null;
            int samples = _state.DatasetOptions.SamplesPerClass * network.ClassCount;
            // Forward + backward ≈ 3× forward FLOPs; ~50 GFLOPS sustained on a 4-core AVX2 laptop,
            // capped by the augmentation pipeline (~25k samples/s).
            double perSecond = Math.Min(25_000, 50e9 / (3.0 * network.FlopsPerSample));
            Summary = $"{network.ParameterCount:N0} parameters · {network.FlopsPerSample / 1e6:0.##} MFLOP per sample · " +
                      $"≈{samples / perSecond:0.#} s per epoch of {samples:N0} samples (rough estimate)";
        }
        catch (ArgumentException ex)
        {
            Preview = null;
            Error = ex.Message;
            Summary = "";
        }
    }

    [RelayCommand]
    private void AddLayer()
    {
        int units = Layers.Count > 0 ? Math.Max(16, Layers[^1].Units / 2) : 256;
        Layers.Add(new HiddenLayerItem { Units = units });
    }

    [RelayCommand]
    private void RemoveLayer(HiddenLayerItem item) => Layers.Remove(item);

    [RelayCommand]
    private void MoveUp(HiddenLayerItem item)
    {
        int i = Layers.IndexOf(item);
        if (i > 0)
            Layers.Move(i, i - 1);
    }

    [RelayCommand]
    private void MoveDown(HiddenLayerItem item)
    {
        int i = Layers.IndexOf(item);
        if (i >= 0 && i < Layers.Count - 1)
            Layers.Move(i, i + 1);
    }

    /// <summary>Replaces the current network with a freshly initialized one built from these settings.</summary>
    [RelayCommand]
    private void Build()
    {
        if (_state.IsTraining)
        {
            MessageBox.Show("Stop training before building a new network.", "OcrNet");
            return;
        }
        if (_state.Network is { } current && current.Metadata.History.Count > 0 && _state.IsDirty &&
            MessageBox.Show("The current network has unsaved training. Replace it with a new, untrained network?",
                "OcrNet", MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes)
            return;

        var metadata = new ModelMetadata { Name = ModelName, Description = Description };
        metadata.Preprocessing = _state.DatasetOptions.Normalizer.ToDictionary();
        var network = NetworkBuilder.Build(CreateSpec(), metadata);
        _state.SetNetwork(network, optimizer: null, path: null);
        _state.Status = $"Built {network}";
    }

    /// <summary>Shows a loaded or newly built network's settings in the designer.</summary>
    private void LoadFromNetwork()
    {
        if (_state.Network is not { } network)
            return;
        _loading = true;
        ModelName = network.Metadata.Name;
        Description = network.Metadata.Description;
        var set = CharacterSet.FromLabels(network.Labels);
        if (set.Name == "Custom")
        {
            SelectedCharacterSetName = CustomSetName;
            CustomCharacters = string.Concat(network.Labels);
        }
        else
        {
            SelectedCharacterSetName = set.Name;
        }
        Layers.Clear();
        foreach (var spec in NetworkBuilder.DescribeHiddenLayers(network))
            Layers.Add(HiddenLayerItem.From(spec));
        _loading = false;
        UpdatePreview();
    }

    partial void OnModelNameChanged(string value)
    {
        if (!_loading && _state.Network is { } network && network.Metadata.Name != value)
        {
            network.Metadata.Name = value;
            _state.IsDirty = true;
        }
    }

    partial void OnDescriptionChanged(string value)
    {
        if (!_loading && _state.Network is { } network && network.Metadata.Description != value)
        {
            network.Metadata.Description = value;
            _state.IsDirty = true;
        }
    }
}
