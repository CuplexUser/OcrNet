using System.Windows;
using System.Windows.Controls;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using OcrNet.App.Services;
using OcrNet.App.ViewModels;

namespace OcrNet.App.Views;

public partial class RecognizeView : UserControl
{
    private Window? _window;

    public RecognizeView()
    {
        InitializeComponent();
        // StrokesChanged covers drawing, erasing and clearing. Subscribed once here: Loaded fires
        // again every time the tab is reselected.
        Canvas.Strokes.StrokesChanged += (_, _) => Recognize();
        Loaded += (_, _) =>
        {
            ApplyPen();
            // Esc is handled at window level: drawing with the mouse does not move keyboard focus
            // into this view, so a KeyDown handler here would rarely see the key.
            _window = Window.GetWindow(this);
            if (_window is not null)
                _window.KeyDown += OnWindowKeyDown;
        };
        Unloaded += (_, _) =>
        {
            if (_window is not null)
                _window.KeyDown -= OnWindowKeyDown;
            _window = null;
        };
    }

    private RecognizeViewModel? ViewModel => DataContext as RecognizeViewModel;

    /// <summary>Number of strokes on the canvas (used by the snapshot smoke test).</summary>
    public int StrokeCount => Canvas.Strokes.Count;

    private void ApplyPen()
    {
        double width = ViewModel?.PenWidth ?? 18;
        Canvas.DefaultDrawingAttributes = new DrawingAttributes
        {
            Color = Color.FromRgb(0xE6, 0xEA, 0xF0),
            Width = width,
            Height = width,
            FitToCurve = true,
            StylusTip = StylusTip.Ellipse,
        };
    }

    private void Recognize()
    {
        if (ViewModel is not { } vm)
            return;
        if (Canvas.Strokes.Count == 0)
        {
            vm.Recognize(null);
            return;
        }
        vm.Recognize(InkRasterizer.Rasterize(Canvas.Strokes, Canvas.ActualWidth, Canvas.ActualHeight));
    }

    private void OnClear(object sender, RoutedEventArgs e) => Canvas.Strokes.Clear();

    private void OnPenWidthChanged(object sender, RoutedPropertyChangedEventArgs<double> e) => ApplyPen();

    private void OnSingleMode(object sender, RoutedEventArgs e)
    {
        if (ViewModel is { } vm)
            vm.LineMode = false;
    }

    /// <summary>
    /// Writes <paramref name="text"/> onto the canvas as strokes traced from a handwriting font
    /// (used by the snapshot smoke test to exercise recognition without a pen).
    /// </summary>
    public void DrawText(string text)
    {
        Canvas.Strokes.Clear();
        var formatted = new FormattedText(text, System.Globalization.CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
            new Typeface("Segoe Print"), 200, Brushes.White, 1.0);
        var geometry = formatted.BuildGeometry(new Point(0, 0));
        var bounds = geometry.Bounds;
        if (bounds.IsEmpty)
            return;
        double scale = Math.Min(Canvas.ActualWidth * 0.7 / bounds.Width, Canvas.ActualHeight * 0.7 / bounds.Height);
        double ox = (Canvas.ActualWidth - bounds.Width * scale) / 2 - bounds.X * scale;
        double oy = (Canvas.ActualHeight - bounds.Height * scale) / 2 - bounds.Y * scale;
        var flat = geometry.GetFlattenedPathGeometry(0.5, ToleranceType.Absolute);
        var strokes = new StrokeCollection();
        foreach (var figure in flat.Figures)
        {
            var points = new StylusPointCollection { new StylusPoint(figure.StartPoint.X * scale + ox, figure.StartPoint.Y * scale + oy) };
            foreach (var segment in figure.Segments)
            {
                var segmentPoints = segment switch
                {
                    PolyLineSegment poly => poly.Points,
                    LineSegment line => [line.Point],
                    _ => new PointCollection(),
                };
                foreach (var p in segmentPoints)
                    points.Add(new StylusPoint(p.X * scale + ox, p.Y * scale + oy));
            }
            var attributes = Canvas.DefaultDrawingAttributes.Clone();
            attributes.Width = attributes.Height = 6;
            strokes.Add(new Stroke(points, attributes));
        }
        Canvas.Strokes.Add(strokes);
    }

    private void OnWindowKeyDown(object sender, KeyEventArgs e)
    {
        // Bubbling event: an open drop-down that closes on Esc has already marked it handled.
        if (e.Key == Key.Escape && !e.Handled && IsVisible)
        {
            Canvas.Strokes.Clear();
            e.Handled = true;
        }
    }
}
