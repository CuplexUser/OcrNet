using System.Globalization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using OcrNet.Data.Datasets;

namespace OcrNet.App.Controls;

/// <summary>
/// Heatmap of [expected, predicted] counts, row-normalized so every class is comparable. The
/// diagonal (correct) uses the signal color, errors use coral; hovering a cell shows its counts.
/// </summary>
public sealed class ConfusionMatrixControl : FrameworkElement
{
    public static readonly DependencyProperty MatrixProperty = DependencyProperty.Register(
        nameof(Matrix), typeof(int[,]), typeof(ConfusionMatrixControl),
        new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty LabelsProperty = DependencyProperty.Register(
        nameof(Labels), typeof(IReadOnlyList<string>), typeof(ConfusionMatrixControl),
        new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

    private static readonly Typeface Face = new("Segoe UI");
    private Point? _hover;

    public int[,]? Matrix
    {
        get => (int[,]?)GetValue(MatrixProperty);
        set => SetValue(MatrixProperty, value);
    }

    public IReadOnlyList<string>? Labels
    {
        get => (IReadOnlyList<string>?)GetValue(LabelsProperty);
        set => SetValue(LabelsProperty, value);
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        _hover = e.GetPosition(this);
        InvalidateVisual();
    }

    protected override void OnMouseLeave(MouseEventArgs e)
    {
        _hover = null;
        InvalidateVisual();
    }

    protected override void OnRender(DrawingContext dc)
    {
        double w = ActualWidth, h = ActualHeight;
        dc.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, w, h));
        var matrix = Matrix;
        var labels = Labels;
        double dpi = VisualTreeHelper.GetDpi(this).PixelsPerDip;
        var muted = (Brush)FindResource("TextMutedBrush");
        if (matrix is null || labels is null || labels.Count != matrix.GetLength(0))
        {
            dc.DrawText(Text("The confusion matrix appears after the first epoch.", 12, muted, dpi), new Point(0, 0));
            return;
        }

        int n = labels.Count;
        const double margin = 22;
        double cell = Math.Max(2, Math.Floor(Math.Min(w - margin, h - margin - 20) / n));
        double gx = margin, gy = margin;
        var signal = (Color)FindResource("SignalColor");
        var negative = (Color)FindResource("NegativeColor");
        double labelSize = Math.Clamp(cell * 0.75, 6, 11);

        for (int i = 0; i < n; i++)
        {
            int rowTotal = 0;
            for (int j = 0; j < n; j++)
                rowTotal += matrix[i, j];
            for (int j = 0; j < n; j++)
            {
                double fraction = rowTotal > 0 ? (double)matrix[i, j] / rowTotal : 0;
                if (fraction <= 0)
                    continue;
                var color = i == j ? signal : negative;
                // Errors are rare, so boost them visually with a square-root ramp.
                double alpha = i == j ? fraction : Math.Sqrt(fraction);
                var brush = new SolidColorBrush(Color.FromArgb((byte)(40 + 215 * alpha), color.R, color.G, color.B));
                dc.DrawRectangle(brush, null, new Rect(gx + j * cell, gy + i * cell, cell - (cell > 4 ? 1 : 0), cell - (cell > 4 ? 1 : 0)));
            }
        }

        if (cell >= 7)
        {
            for (int i = 0; i < n; i++)
            {
                var label = Text(CharacterSet.Display(labels[i]), labelSize, muted, dpi);
                dc.DrawText(label, new Point(gx - label.Width - 4, gy + i * cell + (cell - label.Height) / 2));
                dc.DrawText(label, new Point(gx + i * cell + (cell - label.Width) / 2, gy - label.Height - 2));
            }
        }

        string caption = "rows: expected · columns: predicted";
        if (_hover is { } p)
        {
            int col = (int)((p.X - gx) / cell), row = (int)((p.Y - gy) / cell);
            if (row >= 0 && row < n && col >= 0 && col < n)
            {
                int rowTotal = 0;
                for (int j = 0; j < n; j++)
                    rowTotal += matrix[row, j];
                caption = $"expected {CharacterSet.Display(labels[row])} → predicted {CharacterSet.Display(labels[col])}: " +
                          $"{matrix[row, col]} of {rowTotal} ({(rowTotal > 0 ? (double)matrix[row, col] / rowTotal : 0):P1})";
                dc.DrawRectangle(null, new Pen(Brushes.White, 1), new Rect(gx + col * cell, gy + row * cell, cell, cell));
            }
        }
        dc.DrawText(Text(caption, 11, muted, dpi), new Point(gx, gy + n * cell + 6));
    }

    private static FormattedText Text(string text, double size, Brush brush, double dpi) =>
        new(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Face, size, brush, dpi);
}
