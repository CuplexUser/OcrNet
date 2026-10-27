using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OcrNet.App.Services;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Data.Datasets;

namespace OcrNet.App.Controls;

/// <summary>
/// Draws the network as columns of neurons. Wide layers are sampled (evenly, or the most active
/// neurons for the current input). First-layer neurons show their receptive field — the 28×28
/// weight pattern they respond to. Edges are colored by weight sign; with an input present their
/// opacity follows the signal actually flowing (|weight × activation|), otherwise |weight|.
/// </summary>
public sealed class NetworkGraphControl : FrameworkElement
{
    public static readonly DependencyProperty NetworkProperty = Register<NeuralNetwork?>(nameof(Network), null);
    public static readonly DependencyProperty InputProperty = Register<float[]?>(nameof(Input), null);
    public static readonly DependencyProperty LayerOutputsProperty = Register<float[][]?>(nameof(LayerOutputs), null);
    public static readonly DependencyProperty MaxNodesProperty = Register(nameof(MaxNodes), 16);
    public static readonly DependencyProperty EdgesPerNodeProperty = Register(nameof(EdgesPerNode), 6);
    public static readonly DependencyProperty ShowMostActiveProperty = Register(nameof(ShowMostActive), true);
    public static readonly DependencyProperty WeightsVersionProperty = Register(nameof(WeightsVersion), 0);

    private static readonly Typeface Face = new("Segoe UI");
    private readonly Dictionary<(int Layer, int Unit), BitmapSource> _receptiveFields = [];
    private (NeuralNetwork? Network, int Version) _fieldsKey;

    public NetworkGraphControl()
    {
        // Show 28×28 images as crisp pixels rather than blurred blobs.
        RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
    }

    private static DependencyProperty Register<T>(string name, T defaultValue) =>
        DependencyProperty.Register(name, typeof(T), typeof(NetworkGraphControl),
            new FrameworkPropertyMetadata(defaultValue, FrameworkPropertyMetadataOptions.AffectsRender));

    public NeuralNetwork? Network { get => (NeuralNetwork?)GetValue(NetworkProperty); set => SetValue(NetworkProperty, value); }

    /// <summary>The normalized input image (28×28) currently being classified.</summary>
    public float[]? Input { get => (float[]?)GetValue(InputProperty); set => SetValue(InputProperty, value); }

    /// <summary>Per-layer outputs from <see cref="NeuralNetwork.PredictDetailed"/>.</summary>
    public float[][]? LayerOutputs { get => (float[][]?)GetValue(LayerOutputsProperty); set => SetValue(LayerOutputsProperty, value); }

    public int MaxNodes { get => (int)GetValue(MaxNodesProperty); set => SetValue(MaxNodesProperty, value); }

    public int EdgesPerNode { get => (int)GetValue(EdgesPerNodeProperty); set => SetValue(EdgesPerNodeProperty, value); }

    public bool ShowMostActive { get => (bool)GetValue(ShowMostActiveProperty); set => SetValue(ShowMostActiveProperty, value); }

    /// <summary>Increment after training so cached receptive-field images are rebuilt.</summary>
    public int WeightsVersion { get => (int)GetValue(WeightsVersionProperty); set => SetValue(WeightsVersionProperty, value); }

    /// <summary>A dense layer plus the layers that belong to its block (batch norm, activation, dropout).</summary>
    private sealed record Column(DenseLayer Dense, int OutputLayerIndex, string Title, bool IsOutput);

    private static List<Column> Columns(NeuralNetwork network)
    {
        var columns = new List<Column>();
        var layers = network.Layers;
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is not DenseLayer dense)
                continue;
            int last = i;
            string activation = "";
            for (int j = i + 1; j < layers.Count && layers[j] is not DenseLayer; j++)
            {
                last = j;
                if (layers[j] is ActivationLayer a)
                    activation = a.Activation.ToString();
            }
            bool isOutput = i == layers.Count - 1;
            string title = isOutput ? $"Output · {dense.Units}" : $"Dense {dense.Units} · {activation}";
            columns.Add(new Column(dense, last, title, isOutput));
        }
        return columns;
    }

    protected override void OnRender(DrawingContext dc)
    {
        double w = ActualWidth, h = ActualHeight;
        dc.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, w, h));
        double dpi = VisualTreeHelper.GetDpi(this).PixelsPerDip;
        var muted = (Brush)FindResource("TextMutedBrush");
        var network = Network;
        if (network is null)
        {
            dc.DrawText(Text("Create or open a network to see it here.", 13, muted, dpi), new Point(8, 8));
            return;
        }

        var columns = Columns(network);
        var outputs = LayerOutputs;
        bool live = outputs is not null && outputs.Length == network.Layers.Count;
        var signal = (Color)FindResource("SignalColor");
        var negative = (Color)FindResource("NegativeColor");
        var highlight = (Brush)FindResource("HighlightBrush");
        var nodeIdle = new SolidColorBrush(Color.FromRgb(0x2A, 0x31, 0x3B));
        var outline = new Pen(new SolidColorBrush(Color.FromRgb(0x4A, 0x55, 0x63)), 1);

        const double top = 36, bottom = 28;
        double usableH = Math.Max(50, h - top - bottom);
        double inputBox = Math.Min(140, usableH * 0.5);
        double inputX = 16;
        double firstColumnX = inputX + inputBox + 70;
        double lastColumnX = w - 90;
        double spacingX = columns.Count > 1 ? (lastColumnX - firstColumnX) / (columns.Count - 1) : 0;

        // Input column: the normalized image.
        var inputSize = network.InputShape;
        double inputY = top + (usableH - inputBox) / 2;
        dc.DrawText(Text($"Input {inputSize.H}×{inputSize.W}", 12, muted, dpi), new Point(inputX, 8));
        var inputRect = new Rect(inputX, inputY, inputBox, inputBox);
        dc.DrawRectangle(nodeIdle, outline, inputRect);
        if (Input is { } input && input.Length == inputSize.Size)
        {
            dc.DrawImage(BitmapFactory.Grayscale(input, inputSize.W, inputSize.H), inputRect);
        }
        else
        {
            var hint = Text("draw on the\nRecognize tab", 11, muted, dpi);
            dc.DrawText(hint, new Point(inputRect.X + (inputBox - hint.Width) / 2, inputRect.Y + (inputBox - hint.Height) / 2));
        }

        // Pick visible units per column and lay them out.
        var visible = new List<int[]>();
        var positions = new List<Point[]>();
        for (int c = 0; c < columns.Count; c++)
        {
            var col = columns[c];
            int units = col.Dense.Units;
            int limit = col.IsOutput ? Math.Max(MaxNodes, Math.Min(units, 64)) : MaxNodes;
            float[]? act = live ? outputs![col.OutputLayerIndex] : null;
            int[] chosen;
            if (units <= limit)
                chosen = [.. Enumerable.Range(0, units)];
            else if (act is not null && ShowMostActive)
                chosen = [.. Enumerable.Range(0, units).OrderByDescending(i => act[i]).Take(limit).Order()];
            else
                chosen = [.. Enumerable.Range(0, limit).Select(i => (int)((long)i * units / limit))];
            visible.Add(chosen);

            double x = columns.Count == 1 ? (firstColumnX + lastColumnX) / 2 : firstColumnX + c * spacingX;
            double step = Math.Min(30, usableH / Math.Max(1, chosen.Length));
            double startY = top + (usableH - step * (chosen.Length - 1)) / 2;
            positions.Add([.. chosen.Select((_, k) => new Point(x, startY + k * step))]);
        }

        // Edges between dense columns.
        for (int c = 1; c < columns.Count; c++)
        {
            var weights = columns[c].Dense.Weights;
            int inputs = columns[c].Dense.Inputs;
            float[]? sourceAct = live ? outputs![columns[c - 1].OutputLayerIndex] : null;
            var edges = new List<(Point From, Point To, float Strength, bool Positive)>();
            for (int t = 0; t < visible[c].Length; t++)
            {
                int j = visible[c][t];
                var candidates = visible[c - 1].Select((i, s) =>
                {
                    float wv = weights[j * inputs + i];
                    float strength = sourceAct is not null ? MathF.Abs(wv * sourceAct[i]) : MathF.Abs(wv);
                    return (s, wv, strength);
                }).OrderByDescending(e => e.strength).Take(EdgesPerNode);
                foreach (var (s, wv, strength) in candidates)
                    edges.Add((positions[c - 1][s], positions[c][t], strength, wv >= 0));
            }
            float maxStrength = edges.Count > 0 ? Math.Max(1e-9f, edges.Max(e => e.Strength)) : 1f;
            foreach (var e in edges.OrderBy(e => e.Strength))
            {
                double alpha = Math.Pow(e.Strength / maxStrength, 0.7);
                var color = e.Positive ? signal : negative;
                var pen = new Pen(new SolidColorBrush(Color.FromArgb((byte)(20 + 200 * alpha), color.R, color.G, color.B)), 0.6 + 1.6 * alpha);
                dc.DrawLine(pen, e.From, e.To);
            }
        }

        // Receptive fields of the first dense layer, between the input and its neurons.
        if (columns.Count > 0)
        {
            EnsureFieldCache(network);
            var first = columns[0];
            double fieldSize = Math.Clamp(usableH / Math.Max(1, visible[0].Length) - 3, 10, 26);
            for (int k = 0; k < visible[0].Length; k++)
            {
                var p = positions[0][k];
                var field = ReceptiveField(first.Dense, visible[0][k], network.InputShape.W, network.InputShape.H);
                var rect = new Rect(p.X - fieldSize - 14, p.Y - fieldSize / 2, fieldSize, fieldSize);
                dc.DrawLine(new Pen(nodeIdle, 1), new Point(inputRect.Right, inputRect.Y + inputBox / 2), new Point(rect.Left, p.Y));
                dc.DrawImage(field, rect);
            }
        }

        // Nodes, titles and output labels.
        for (int c = 0; c < columns.Count; c++)
        {
            var col = columns[c];
            float[]? act = live ? outputs![col.OutputLayerIndex] : null;
            float maxAct = act is null ? 1f : Math.Max(1e-6f, visible[c].Max(i => MathF.Abs(act[i])));
            int top1 = act is not null && col.IsOutput ? Array.IndexOf(act, act.Max()) : -1;
            double step = visible[c].Length > 1 ? positions[c][1].Y - positions[c][0].Y : 30;
            double radius = Math.Clamp(step * 0.38, 2.5, 9);

            var title = Text(col.Title, 12, muted, dpi);
            dc.DrawText(title, new Point(Math.Clamp(positions[c][0].X - title.Width / 2, 0, w - title.Width), 8));

            for (int k = 0; k < visible[c].Length; k++)
            {
                int unit = visible[c][k];
                var p = positions[c][k];
                Brush fill = nodeIdle;
                if (act is not null)
                {
                    float a = Math.Clamp(act[unit] / maxAct, 0f, 1f);
                    fill = new SolidColorBrush(Color.FromRgb(
                        (byte)(0x2A + (signal.R - 0x2A) * a), (byte)(0x31 + (signal.G - 0x31) * a), (byte)(0x3B + (signal.B - 0x3B) * a)));
                }
                dc.DrawEllipse(fill, unit == top1 ? new Pen(highlight, 2) : outline, p, radius, radius);

                if (col.IsOutput && step >= 7)
                {
                    string label = CharacterSet.Display(network.Labels[unit]);
                    if (act is not null)
                        label += $"  {act[unit]:P0}";
                    var text = Text(label, Math.Clamp(step * 0.7, 7, 12), unit == top1 ? highlight : muted, dpi);
                    dc.DrawText(text, new Point(p.X + radius + 6, p.Y - text.Height / 2));
                }
            }

            int hidden = col.Dense.Units - visible[c].Length;
            if (hidden > 0)
            {
                var more = Text($"+{hidden:N0} more", 10, muted, dpi);
                dc.DrawText(more, new Point(positions[c][0].X - more.Width / 2, h - bottom + 6));
            }
        }
    }

    private void EnsureFieldCache(NeuralNetwork network)
    {
        if (_fieldsKey.Network == network && _fieldsKey.Version == WeightsVersion)
            return;
        _receptiveFields.Clear();
        _fieldsKey = (network, WeightsVersion);
    }

    private BitmapSource ReceptiveField(DenseLayer dense, int unit, int width, int height)
    {
        if (_receptiveFields.TryGetValue((0, unit), out var cached))
            return cached;
        var bitmap = BitmapFactory.Diverging(dense.Weights.AsSpan(unit * dense.Inputs, width * height), width, height);
        _receptiveFields[(0, unit)] = bitmap;
        return bitmap;
    }

    private static FormattedText Text(string text, double size, Brush brush, double dpi) =>
        new(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Face, size, brush, dpi);
}
