using System.Collections;
using System.Collections.Specialized;
using System.Globalization;
using System.Windows;
using System.Windows.Media;

namespace OcrNet.App.Controls;

/// <summary>One line of a <see cref="LineChart"/>.</summary>
public sealed class ChartSeries
{
    public required string Name { get; init; }

    public required Brush Stroke { get; init; }

    public bool Dashed { get; init; }

    public List<Point> Points { get; } = [];
}

/// <summary>
/// Lightweight multi-series line chart drawn directly with a DrawingContext: light grid,
/// auto-scaled axes, and a legend. Redraws when <see cref="Series"/> or the collection changes.
/// </summary>
public sealed class LineChart : FrameworkElement
{
    public static readonly DependencyProperty SeriesProperty = DependencyProperty.Register(
        nameof(Series), typeof(IEnumerable), typeof(LineChart),
        new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender, OnSeriesChanged));

    public static readonly DependencyProperty TitleProperty = DependencyProperty.Register(
        nameof(Title), typeof(string), typeof(LineChart),
        new FrameworkPropertyMetadata("", FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty YFormatProperty = DependencyProperty.Register(
        nameof(YFormat), typeof(string), typeof(LineChart),
        new FrameworkPropertyMetadata("0.###", FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty YMinProperty = DependencyProperty.Register(
        nameof(YMin), typeof(double), typeof(LineChart),
        new FrameworkPropertyMetadata(double.NaN, FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty YMaxProperty = DependencyProperty.Register(
        nameof(YMax), typeof(double), typeof(LineChart),
        new FrameworkPropertyMetadata(double.NaN, FrameworkPropertyMetadataOptions.AffectsRender));

    private static readonly Typeface LabelFace = new("Segoe UI");
    private static readonly Brush GridBrush = Frozen(new SolidColorBrush(Color.FromArgb(0x30, 0x8B, 0x95, 0xA3)));
    private static readonly Brush LabelBrush = Frozen(new SolidColorBrush(Color.FromRgb(0x8B, 0x95, 0xA3)));

    public IEnumerable? Series
    {
        get => (IEnumerable?)GetValue(SeriesProperty);
        set => SetValue(SeriesProperty, value);
    }

    public string Title
    {
        get => (string)GetValue(TitleProperty);
        set => SetValue(TitleProperty, value);
    }

    public string YFormat
    {
        get => (string)GetValue(YFormatProperty);
        set => SetValue(YFormatProperty, value);
    }

    /// <summary>Fixed lower bound of the y axis (NaN = automatic).</summary>
    public double YMin
    {
        get => (double)GetValue(YMinProperty);
        set => SetValue(YMinProperty, value);
    }

    /// <summary>Fixed upper bound of the y axis (NaN = automatic).</summary>
    public double YMax
    {
        get => (double)GetValue(YMaxProperty);
        set => SetValue(YMaxProperty, value);
    }

    private static void OnSeriesChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        var chart = (LineChart)d;
        if (e.OldValue is INotifyCollectionChanged oldCollection)
            oldCollection.CollectionChanged -= chart.OnCollectionChanged;
        if (e.NewValue is INotifyCollectionChanged newCollection)
            newCollection.CollectionChanged += chart.OnCollectionChanged;
    }

    private void OnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e) => InvalidateVisual();

    /// <summary>Call after appending points to a series in place.</summary>
    public void Refresh() => InvalidateVisual();

    protected override void OnRender(DrawingContext dc)
    {
        double w = ActualWidth, h = ActualHeight;
        dc.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, w, h));
        var series = Series?.OfType<ChartSeries>().ToList() ?? [];
        double dpi = VisualTreeHelper.GetDpi(this).PixelsPerDip;

        const double left = 48, right = 12, top = 26, bottom = 22;
        var plot = new Rect(left, top, Math.Max(1, w - left - right), Math.Max(1, h - top - bottom));

        if (!string.IsNullOrEmpty(Title))
            dc.DrawText(Text(Title, 12, Brushes.White, dpi), new Point(0, 0));

        var points = series.SelectMany(s => s.Points).Where(p => double.IsFinite(p.Y)).ToList();
        if (points.Count == 0)
        {
            var empty = Text("No data yet", 12, LabelBrush, dpi);
            dc.DrawText(empty, new Point(plot.Left + (plot.Width - empty.Width) / 2, plot.Top + (plot.Height - empty.Height) / 2));
            return;
        }

        double xMin = points.Min(p => p.X), xMax = points.Max(p => p.X);
        if (xMax - xMin < 1e-9) xMax = xMin + 1;
        double yMin = double.IsNaN(YMin) ? points.Min(p => p.Y) : YMin;
        double yMax = double.IsNaN(YMax) ? points.Max(p => p.Y) : YMax;
        if (yMax - yMin < 1e-9)
        {
            yMax += 0.5;
            yMin -= 0.5;
        }
        else if (double.IsNaN(YMin) || double.IsNaN(YMax))
        {
            double pad = (yMax - yMin) * 0.08;
            if (double.IsNaN(YMin)) yMin -= pad;
            if (double.IsNaN(YMax)) yMax += pad;
        }

        Point Map(Point p) => new(
            plot.Left + (p.X - xMin) / (xMax - xMin) * plot.Width,
            plot.Bottom - (p.Y - yMin) / (yMax - yMin) * plot.Height);

        // Grid and y labels.
        var gridPen = new Pen(GridBrush, 1);
        for (int i = 0; i <= 4; i++)
        {
            double v = yMin + (yMax - yMin) * i / 4;
            double y = plot.Bottom - plot.Height * i / 4;
            dc.DrawLine(gridPen, new Point(plot.Left, y), new Point(plot.Right, y));
            var label = Text(v.ToString(YFormat, CultureInfo.InvariantCulture), 10, LabelBrush, dpi);
            dc.DrawText(label, new Point(plot.Left - label.Width - 6, y - label.Height / 2));
        }

        // X labels (epochs): first, last and a middle tick.
        foreach (double v in new[] { xMin, (xMin + xMax) / 2, xMax }.Distinct())
        {
            var label = Text(v.ToString("0.#", CultureInfo.InvariantCulture), 10, LabelBrush, dpi);
            double x = Map(new Point(v, yMin)).X;
            dc.DrawText(label, new Point(Math.Clamp(x - label.Width / 2, plot.Left, plot.Right - label.Width), plot.Bottom + 4));
        }

        // Series.
        foreach (var s in series)
        {
            var valid = s.Points.Where(p => double.IsFinite(p.Y)).Select(Map).ToList();
            if (valid.Count == 0)
                continue;
            var pen = new Pen(s.Stroke, 2) { LineJoin = PenLineJoin.Round };
            if (s.Dashed)
                pen.DashStyle = new DashStyle([3, 2], 0);
            var geometry = new StreamGeometry();
            using (var ctx = geometry.Open())
            {
                ctx.BeginFigure(valid[0], false, false);
                ctx.PolyLineTo(valid.Skip(1).ToList(), true, true);
            }
            geometry.Freeze();
            dc.DrawGeometry(null, pen, geometry);
            if (valid.Count <= 40)
                foreach (var p in valid)
                    dc.DrawEllipse(s.Stroke, null, p, 2.5, 2.5);
        }

        // Legend, right-aligned on the title row.
        double lx = w - right;
        foreach (var s in Enumerable.Reverse(series))
        {
            var label = Text(s.Name, 11, LabelBrush, dpi);
            lx -= label.Width;
            dc.DrawText(label, new Point(lx, 2));
            lx -= 14;
            dc.DrawRectangle(s.Stroke, null, new Rect(lx, 8, 10, 3));
            lx -= 12;
        }
    }

    private static FormattedText Text(string text, double size, Brush brush, double dpi) =>
        new(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, LabelFace, size, brush, dpi);

    private static Brush Frozen(Brush brush)
    {
        brush.Freeze();
        return brush;
    }
}
