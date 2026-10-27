using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace OcrNet.App.Services;

/// <summary>Converts float arrays into small bitmaps for previews and heatmaps.</summary>
public static class BitmapFactory
{
    /// <summary>Ink (0..1) rendered light-on-dark.</summary>
    public static BitmapSource Grayscale(ReadOnlySpan<float> pixels, int width, int height)
    {
        var bytes = new byte[width * height];
        for (int i = 0; i < bytes.Length; i++)
            bytes[i] = (byte)Math.Clamp((int)(pixels[i] * 255f), 0, 255);
        var bitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, null, bytes, width);
        bitmap.Freeze();
        return bitmap;
    }

    /// <summary>Signed values: teal for positive, coral for negative, dark for zero. Scaled by the max |value|.</summary>
    public static BitmapSource Diverging(ReadOnlySpan<float> values, int width, int height)
    {
        float max = 1e-6f;
        foreach (float v in values)
            max = MathF.Max(max, MathF.Abs(v));
        var positive = (Color)Application.Current.FindResource("SignalColor");
        var negative = (Color)Application.Current.FindResource("NegativeColor");
        var zero = Color.FromRgb(0x14, 0x18, 0x1E);

        var bytes = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            float t = values[i] / max;
            var target = t >= 0 ? positive : negative;
            float a = MathF.Min(1f, MathF.Abs(t));
            bytes[i * 4 + 0] = Lerp(zero.B, target.B, a);
            bytes[i * 4 + 1] = Lerp(zero.G, target.G, a);
            bytes[i * 4 + 2] = Lerp(zero.R, target.R, a);
            bytes[i * 4 + 3] = 255;
        }
        var bitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Bgra32, null, bytes, width * 4);
        bitmap.Freeze();
        return bitmap;
    }

    private static byte Lerp(byte from, byte to, float t) => (byte)(from + (to - from) * t);
}
