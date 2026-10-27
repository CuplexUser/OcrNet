using System.Globalization;
using System.Windows;
using System.Windows.Media;

namespace OcrNet.App.Controls;

/// <summary>Histogram of a set of values (e.g. the weights of one layer), symmetric around zero.</summary>
public sealed class HistogramControl : FrameworkElement
{
    public static readonly DependencyProperty ValuesProperty = DependencyProperty.Register(
        nameof(Values), typeof(IReadOnlyList<float>), typeof(HistogramControl),
        new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty TitleProperty = DependencyProperty.Register(
        nameof(Title), typeof(string), typeof(HistogramControl),
        new FrameworkPropertyMetadata("", FrameworkPropertyMetadataOptions.AffectsRender));

    private const int Bins = 41;
    private static readonly Typeface Face = new("Segoe UI");

    public IReadOnlyList<float>? Values
    {
        get => (IReadOnlyList<float>?)GetValue(ValuesProperty);
        set => SetValue(ValuesProperty, value);
    }

    public string Title
    {
        get => (string)GetValue(TitleProperty);
        set => SetValue(TitleProperty, value);
    }

    protected override void OnRender(DrawingContext dc)
    {
        double w = ActualWidth, h = ActualHeight;
        dc.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, w, h));
        double dpi = VisualTreeHelper.GetDpi(this).PixelsPerDip;
        var muted = (Brush)FindResource("TextMutedBrush");
        dc.DrawText(Text(Title, 11, muted, dpi), new Point(0, 0));
        var values = Values;
        if (values is null || values.Count == 0)
            return;

        // Clip the range at the 99.5th percentile of |w| so a few outliers do not flatten the plot.
        int stride = Math.Max(1, values.Count / 20_000); // a sample is plenty for the percentile
        var magnitudes = Enumerable.Range(0, (values.Count + stride - 1) / stride).Select(i => MathF.Abs(values[i * stride])).Order().ToArray();
        float range = Math.Max(1e-6f, magnitudes[(int)(magnitudes.Length * 0.995)]);
        var counts = new int[Bins];
        foreach (float v in values)
        {
            int bin = (int)((v / range + 1) / 2 * (Bins - 1) + 0.5);
            counts[Math.Clamp(bin, 0, Bins - 1)]++;
        }
        int max = counts.Max();

        const double top = 18, bottom = 16;
        double plotH = Math.Max(1, h - top - bottom);
        double barW = w / Bins;
        var positive = (Brush)FindResource("SignalBrush");
        var negative = (Brush)FindResource("NegativeBrush");
        for (int i = 0; i < Bins; i++)
        {
            double barH = plotH * counts[i] / max;
            var brush = i < Bins / 2 ? negative : i > Bins / 2 ? positive : muted;
            dc.DrawRectangle(brush, null, new Rect(i * barW + 0.5, top + plotH - barH, Math.Max(1, barW - 1), barH));
        }

        var minLabel = Text((-range).ToString("0.###", CultureInfo.InvariantCulture), 10, muted, dpi);
        var maxLabel = Text(range.ToString("0.###", CultureInfo.InvariantCulture), 10, muted, dpi);
        dc.DrawText(minLabel, new Point(0, h - bottom + 2));
        dc.DrawText(maxLabel, new Point(w - maxLabel.Width, h - bottom + 2));
        var stats = Text($"n={values.Count:N0}  σ={StdDev(values):0.###}", 10, muted, dpi);
        dc.DrawText(stats, new Point(w - stats.Width, 0));
    }

    private static double StdDev(IReadOnlyList<float> values)
    {
        double mean = values.Average(v => (double)v);
        return Math.Sqrt(values.Average(v => (v - mean) * (v - mean)));
    }

    private static FormattedText Text(string text, double size, Brush brush, double dpi) =>
        new(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Face, size, brush, dpi);
}
