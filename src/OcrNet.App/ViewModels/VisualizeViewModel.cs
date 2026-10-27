using System.Collections.ObjectModel;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using OcrNet.App.Services;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;

namespace OcrNet.App.ViewModels;

public sealed record HistogramItem(string Title, float[] Values);

public sealed record FieldItem(BitmapSource Image, string Caption);

/// <summary>Network graph, first-layer receptive fields and weight histograms.</summary>
public sealed partial class VisualizeViewModel : ObservableObject
{
    private readonly AppState _state;

    public VisualizeViewModel(AppState state)
    {
        _state = state;
        state.NetworkReplaced += (_, _) => Refresh(weightsChanged: true);
        state.WeightsChanged += (_, _) => Refresh(weightsChanged: true);
        state.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(AppState.LastInput))
                Refresh(weightsChanged: false);
        };
    }

    [ObservableProperty]
    public partial NeuralNetwork? Network { get; set; }

    [ObservableProperty]
    public partial float[]? Input { get; set; }

    [ObservableProperty]
    public partial float[][]? LayerOutputs { get; set; }

    [ObservableProperty]
    public partial int WeightsVersion { get; set; }

    [ObservableProperty]
    public partial int MaxNodes { get; set; } = 16;

    [ObservableProperty]
    public partial int EdgesPerNode { get; set; } = 6;

    [ObservableProperty]
    public partial bool ShowMostActive { get; set; } = true;

    [ObservableProperty]
    public partial string Architecture { get; set; } = "";

    public ObservableCollection<FieldItem> ReceptiveFields { get; } = [];

    public ObservableCollection<HistogramItem> Histograms { get; } = [];

    /// <summary>Rebuilds everything shown; cheap enough to call once per epoch.</summary>
    public void Refresh(bool weightsChanged)
    {
        var network = _state.Network;
        Network = network;
        Input = _state.LastInput;
        if (network is null)
        {
            LayerOutputs = null;
            ReceptiveFields.Clear();
            Histograms.Clear();
            Architecture = "";
            return;
        }

        Architecture = network.ToString();
        LayerOutputs = Input is { } input && input.Length == network.InputShape.Size
            ? network.PredictDetailed(input).LayerOutputs
            : null;

        if (weightsChanged)
            WeightsVersion++;
        RefreshFields(network);
        if (weightsChanged)
            RefreshHistograms(network);
    }

    partial void OnShowMostActiveChanged(bool value)
    {
        if (Network is { } network)
            RefreshFields(network);
    }

    /// <summary>The 48 first-layer neurons shown: the most active for the current input, else the first ones.</summary>
    private void RefreshFields(NeuralNetwork network)
    {
        ReceptiveFields.Clear();
        if (network.Layers.FirstOrDefault() is not DenseLayer first || network.InputShape.C != 1)
            return;
        int w = network.InputShape.W, h = network.InputShape.H;
        IEnumerable<int> units = Enumerable.Range(0, Math.Min(48, first.Units));
        string caption = "#{0}";
        if (ShowMostActive && LayerOutputs is { } outputs)
        {
            // Activation after the first block's non-linearity.
            int blockEnd = Enumerable.Range(1, network.Layers.Count - 1).TakeWhile(i => network.Layers[i] is not DenseLayer).LastOrDefault();
            var act = outputs[blockEnd];
            units = Enumerable.Range(0, first.Units).OrderByDescending(i => act[i]).Take(48);
            caption = "#{0} · {1:0.00}";
            foreach (int u in units)
                ReceptiveFields.Add(new FieldItem(BitmapFactory.Diverging(first.Weights.AsSpan(u * first.Inputs, w * h), w, h), string.Format(caption, u, act[u])));
            return;
        }
        foreach (int u in units)
            ReceptiveFields.Add(new FieldItem(BitmapFactory.Diverging(first.Weights.AsSpan(u * first.Inputs, w * h), w, h), string.Format(caption, u)));
    }

    private void RefreshHistograms(NeuralNetwork network)
    {
        Histograms.Clear();
        int index = 0;
        foreach (var dense in network.Layers.OfType<DenseLayer>())
        {
            index++;
            float[] copy;
            lock (network.SyncRoot)
                copy = (float[])dense.Weights.Clone();
            string role = dense == network.Layers[^1] ? "output" : $"hidden {index}";
            Histograms.Add(new HistogramItem($"Layer {index} ({role}) weights · {dense.Inputs}→{dense.Units}", copy));
        }
    }
}
