namespace OcrNet.Data.Imaging;

/// <summary>Single-channel float image, row-major; 0 = background, 1 = ink.</summary>
public sealed class GrayImage
{
    public GrayImage(int width, int height)
        : this(width, height, new float[width * height])
    {
    }

    public GrayImage(int width, int height, float[] pixels)
    {
        if (pixels.Length != width * height)
            throw new ArgumentException("Pixel count does not match dimensions.", nameof(pixels));
        Width = width;
        Height = height;
        Pixels = pixels;
    }

    public int Width { get; }

    public int Height { get; }

    public float[] Pixels { get; }

    public float this[int x, int y]
    {
        get => Pixels[y * Width + x];
        set => Pixels[y * Width + x] = value;
    }

    public GrayImage Clone() => new(Width, Height, (float[])Pixels.Clone());

    /// <summary>Bilinear sample with zero outside the image.</summary>
    public float Sample(float x, float y)
    {
        if (x >= 0f && y >= 0f && x < Width - 1 && y < Height - 1)
        {
            // Fast path: all four neighbors are inside the image.
            int ix = (int)x, iy = (int)y;
            float tx = x - ix, ty = y - iy;
            int i = iy * Width + ix;
            float[] p = Pixels;
            float top = p[i] + (p[i + 1] - p[i]) * tx;
            float bottom = p[i + Width] + (p[i + Width + 1] - p[i + Width]) * tx;
            return top + (bottom - top) * ty;
        }
        if (x <= -1f || y <= -1f || x >= Width || y >= Height)
            return 0f;

        int x0 = (int)MathF.Floor(x), y0 = (int)MathF.Floor(y);
        float fx = x - x0, fy = y - y0;
        float p00 = Get(x0, y0), p10 = Get(x0 + 1, y0), p01 = Get(x0, y0 + 1), p11 = Get(x0 + 1, y0 + 1);
        return (p00 * (1 - fx) + p10 * fx) * (1 - fy) + (p01 * (1 - fx) + p11 * fx) * fy;
    }

    private float Get(int x, int y) =>
        (uint)x < (uint)Width && (uint)y < (uint)Height ? Pixels[y * Width + x] : 0f;

    public byte[] ToBytes()
    {
        var bytes = new byte[Pixels.Length];
        for (int i = 0; i < bytes.Length; i++)
            bytes[i] = (byte)Math.Clamp((int)MathF.Round(Pixels[i] * 255f), 0, 255);
        return bytes;
    }

    public static GrayImage FromBytes(int width, int height, ReadOnlySpan<byte> bytes)
    {
        var image = new GrayImage(width, height);
        for (int i = 0; i < image.Pixels.Length; i++)
            image.Pixels[i] = bytes[i] / 255f;
        return image;
    }

    /// <summary>Tight bounding box of pixels above <paramref name="threshold"/>, or null when the image is blank.</summary>
    public (int X, int Y, int Width, int Height)? InkBounds(float threshold = 0.1f)
    {
        int minX = Width, minY = Height, maxX = -1, maxY = -1;
        for (int y = 0; y < Height; y++)
        {
            int row = y * Width;
            for (int x = 0; x < Width; x++)
            {
                if (Pixels[row + x] <= threshold)
                    continue;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
        return maxX < 0 ? null : (minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    public GrayImage Crop(int x, int y, int width, int height)
    {
        var result = new GrayImage(width, height);
        for (int row = 0; row < height; row++)
            Array.Copy(Pixels, (y + row) * Width + x, result.Pixels, row * width, width);
        return result;
    }
}
