using System.Windows;
using System.Windows.Ink;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OcrNet.Data.Imaging;

namespace OcrNet.App.Services;

/// <summary>Turns InkCanvas strokes into a grayscale image (ink = 1) for the normalizer.</summary>
public static class InkRasterizer
{
    /// <summary>Renders the strokes at <paramref name="scale"/> of the canvas size.</summary>
    public static GrayImage Rasterize(StrokeCollection strokes, double canvasWidth, double canvasHeight, double scale = 0.5)
    {
        int width = Math.Max(1, (int)Math.Ceiling(canvasWidth * scale));
        int height = Math.Max(1, (int)Math.Ceiling(canvasHeight * scale));
        var visual = new DrawingVisual();
        using (var dc = visual.RenderOpen())
        {
            dc.DrawRectangle(Brushes.Black, null, new Rect(0, 0, width, height));
            dc.PushTransform(new ScaleTransform(scale, scale));
            foreach (var stroke in strokes)
            {
                var copy = stroke.Clone();
                copy.DrawingAttributes.Color = Colors.White;
                copy.Draw(dc);
            }
            dc.Pop();
        }
        var bitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
        bitmap.Render(visual);
        var pixels = new byte[width * height * 4];
        bitmap.CopyPixels(pixels, width * 4, 0);

        var image = new GrayImage(width, height);
        for (int i = 0; i < image.Pixels.Length; i++)
            image.Pixels[i] = pixels[i * 4 + 2] / 255f;
        return image;
    }
}
