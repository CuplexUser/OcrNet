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
    private bool _erasing;
    private Point _eraseFrom;

    public RecognizeView()
    {
        InitializeComponent();
        // StrokesChanged covers drawing, erasing and clearing. Subscribed once here: Loaded fires
        // again every time the tab is reselected.
        // While erasing, recognition waits until the button is released.
        Canvas.Strokes.StrokesChanged += (_, _) =>
        {
            if (!_erasing)
                Recognize();
        };
        Canvas.PreviewMouseRightButtonDown += OnEraseStart;
        Canvas.PreviewMouseMove += OnEraseMove;
        Canvas.PreviewMouseRightButtonUp += OnEraseEnd;
        Canvas.LostMouseCapture += (_, _) => StopErasing();
        Loaded += (_, _) =>
        {
            ApplyPen();
            // Esc is handled at window level, before any control sees it: keyboard focus may be
            // anywhere (or nowhere) in this view, and focused controls can mark a bubbling KeyDown
            // handled before it reaches the window.
            _window = Window.GetWindow(this);
            if (_window is not null)
                _window.PreviewKeyDown += OnWindowPreviewKeyDown;
        };
        Unloaded += (_, _) =>
        {
            if (_window is not null)
                _window.PreviewKeyDown -= OnWindowPreviewKeyDown;
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

    private void OnWindowPreviewKeyDown(object sender, KeyEventArgs e)
    {
        // An open drop-down gets Esc to close itself.
        if (e.Key == Key.Escape && IsVisible && !IsDropDownOpen(Keyboard.FocusedElement as DependencyObject))
        {
            Canvas.Strokes.Clear();
            e.Handled = true;
        }
    }

    private static bool IsDropDownOpen(DependencyObject? focused) =>
        focused is ComboBox { IsDropDownOpen: true }
        || (focused is not null && ItemsControl.ItemsControlFromItemContainer(focused) is ComboBox { IsDropDownOpen: true });

    // Right-drag erases: the InkCanvas only draws with the left button, so erasing is done here by
    // hit-testing the pointer path against the strokes.

    private double EraserSize => (ViewModel?.PenWidth ?? 18) * 1.5;

    private void OnEraseStart(object sender, MouseButtonEventArgs e)
    {
        if (e.LeftButton == MouseButtonState.Pressed || !Canvas.CaptureMouse())
            return;
        _erasing = true;
        _eraseFrom = e.GetPosition(Canvas);
        Mouse.OverrideCursor = Cursors.Cross;
        EraseTo(_eraseFrom);
        e.Handled = true;
    }

    private void OnEraseMove(object sender, MouseEventArgs e)
    {
        if (!_erasing)
            return;
        EraseTo(e.GetPosition(Canvas));
        e.Handled = true;
    }

    private void OnEraseEnd(object sender, MouseButtonEventArgs e)
    {
        if (!_erasing)
            return;
        Canvas.ReleaseMouseCapture(); // raises LostMouseCapture, which stops erasing
        e.Handled = true;
    }

    private void EraseTo(Point point)
    {
        Canvas.Strokes.Erase([_eraseFrom, point], new EllipseStylusShape(EraserSize, EraserSize));
        _eraseFrom = point;
    }

    private void StopErasing()
    {
        if (!_erasing)
            return;
        _erasing = false;
        Mouse.OverrideCursor = null;
        Recognize();
    }
}
