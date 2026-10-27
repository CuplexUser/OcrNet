using System.Buffers;
using System.Globalization;

namespace OcrNet.Data.Imaging;

/// <summary>Parameters of <see cref="Normalizer"/>; stored with each model so inference matches training.</summary>
public sealed class NormalizerOptions
{
    /// <summary>Side of the square network input (28 like MNIST).</summary>
    public int OutputSize { get; set; } = 28;

    /// <summary>The character's longer side is scaled to this many pixels.</summary>
    public int FitSize { get; set; } = 20;

    /// <summary>Pixels at or below this value count as background when finding the bounding box.</summary>
    public float InkThreshold { get; set; } = 0.1f;

    /// <summary>Shift the result so its center of mass is in the middle (as MNIST does).</summary>
    public bool CenterOfMass { get; set; } = true;

    public Dictionary<string, string> ToDictionary() => new()
    {
        [nameof(OutputSize)] = OutputSize.ToString(CultureInfo.InvariantCulture),
        [nameof(FitSize)] = FitSize.ToString(CultureInfo.InvariantCulture),
        [nameof(InkThreshold)] = InkThreshold.ToString(CultureInfo.InvariantCulture),
        [nameof(CenterOfMass)] = CenterOfMass.ToString(CultureInfo.InvariantCulture),
    };

    public static NormalizerOptions FromDictionary(IReadOnlyDictionary<string, string> values)
    {
        var o = new NormalizerOptions();
        if (values.TryGetValue(nameof(OutputSize), out var s) && int.TryParse(s, CultureInfo.InvariantCulture, out int size)) o.OutputSize = size;
        if (values.TryGetValue(nameof(FitSize), out s) && int.TryParse(s, CultureInfo.InvariantCulture, out int fit)) o.FitSize = fit;
        if (values.TryGetValue(nameof(InkThreshold), out s) && float.TryParse(s, CultureInfo.InvariantCulture, out float t)) o.InkThreshold = t;
        if (values.TryGetValue(nameof(CenterOfMass), out s) && bool.TryParse(s, out bool com)) o.CenterOfMass = com;
        return o;
    }
}

/// <summary>
/// Turns an arbitrary grayscale image of one character into the network input. Used for font
/// glyphs during training and for drawings at recognition time, which is what makes the two
/// domains comparable: crop to the ink, scale the longer side to <see cref="NormalizerOptions.FitSize"/>
/// keeping the aspect ratio, then center by center of mass in an <see cref="NormalizerOptions.OutputSize"/> square.
/// </summary>
public static class Normalizer
{
    public static float[] Normalize(GrayImage source, NormalizerOptions options)
    {
        var output = new float[options.OutputSize * options.OutputSize];
        Normalize(source, options, output);
        return output;
    }

    public static void Normalize(GrayImage source, NormalizerOptions options, Span<float> output)
    {
        int size = options.OutputSize;
        output.Clear();
        var bounds = source.InkBounds(options.InkThreshold);
        if (bounds is null)
            return;
        var (bx, by, bw, bh) = bounds.Value;

        float scale = (float)options.FitSize / Math.Max(bw, bh);
        int tw = Math.Clamp((int)MathF.Round(bw * scale), 1, size);
        int th = Math.Clamp((int)MathF.Round(bh * scale), 1, size);
        int ox = (size - tw) / 2, oy = (size - th) / 2;

        ResampleArea(source, bx, by, bw, bh, tw, th, output, ox, oy, size);

        if (options.CenterOfMass)
            ShiftToCenterOfMass(output, size, ox, oy, tw, th);
    }

    /// <summary>
    /// Exact box-filter (area-average) resampling of the source rectangle (bx, by, bw, bh) to
    /// tw×th, done separably: rows first, then columns. Each output pixel is the mean of the source
    /// area it covers, which avoids aliasing when large drawings are shrunk to 20 pixels.
    /// </summary>
    private static void ResampleArea(GrayImage source, int bx, int by, int bw, int bh, int tw, int th,
        Span<float> output, int ox, int oy, int stride)
    {
        var (xIndex, xWeight, xStart) = AreaWeights(bw, tw);
        var (yIndex, yWeight, yStart) = AreaWeights(bh, th);
        var rows = ArrayPool<float>.Shared.Rent(bh * tw);
        try
        {
            float[] src = source.Pixels;
            for (int r = 0; r < bh; r++)
            {
                int srcRow = (by + r) * source.Width + bx;
                for (int j = 0; j < tw; j++)
                {
                    float sum = 0f;
                    for (int e = xStart[j]; e < xStart[j + 1]; e++)
                        sum += src[srcRow + xIndex[e]] * xWeight[e];
                    rows[r * tw + j] = sum;
                }
            }
            for (int i = 0; i < th; i++)
            {
                var dst = output.Slice((oy + i) * stride + ox, tw);
                for (int j = 0; j < tw; j++)
                {
                    float sum = 0f;
                    for (int e = yStart[i]; e < yStart[i + 1]; e++)
                        sum += rows[yIndex[e] * tw + j] * yWeight[e];
                    dst[j] = Math.Clamp(sum, 0f, 1f);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rows);
        }
    }

    /// <summary>For each of <paramref name="dstLength"/> output cells, the source cells it overlaps and their weights.</summary>
    private static (int[] Index, float[] Weight, int[] Start) AreaWeights(int srcLength, int dstLength)
    {
        float step = (float)srcLength / dstLength;
        int maxPerCell = (int)MathF.Ceiling(step) + 1;
        var index = new int[dstLength * maxPerCell];
        var weight = new float[dstLength * maxPerCell];
        var start = new int[dstLength + 1];
        int e = 0;
        for (int j = 0; j < dstLength; j++)
        {
            start[j] = e;
            float from = j * step, to = from + step;
            int first = (int)MathF.Floor(from);
            int last = Math.Min(srcLength - 1, (int)MathF.Ceiling(to) - 1);
            for (int i = first; i <= last; i++)
            {
                float overlap = MathF.Min(to, i + 1) - MathF.Max(from, i);
                if (overlap <= 0f)
                    continue;
                index[e] = i;
                weight[e] = overlap / step;
                e++;
            }
        }
        start[dstLength] = e;
        return (index, weight, start);
    }

    private static void ShiftToCenterOfMass(Span<float> image, int size, int ox, int oy, int tw, int th)
    {
        double mass = 0, mx = 0, my = 0;
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float v = image[y * size + x];
                mass += v;
                mx += v * x;
                my += v * y;
            }
        if (mass <= 0)
            return;

        float center = (size - 1) / 2f;
        int dx = (int)Math.Round(center - mx / mass);
        int dy = (int)Math.Round(center - my / mass);
        // Never push ink outside the frame.
        dx = Math.Clamp(dx, -ox, size - (ox + tw));
        dy = Math.Clamp(dy, -oy, size - (oy + th));
        if (dx == 0 && dy == 0)
            return;

        Span<float> copy = stackalloc float[size * size];
        image.CopyTo(copy);
        image.Clear();
        for (int y = 0; y < size; y++)
        {
            int ny = y + dy;
            if ((uint)ny >= (uint)size)
                continue;
            for (int x = 0; x < size; x++)
            {
                int nx = x + dx;
                if ((uint)nx < (uint)size)
                    image[ny * size + nx] = copy[y * size + x];
            }
        }
    }

    /// <summary>
    /// Scales an image so its ink fits a <paramref name="fit"/>-pixel box centered in a
    /// <paramref name="size"/>² canvas. Drawings are stored this way so they can go through the
    /// same augmentation path as rendered glyphs.
    /// </summary>
    public static GrayImage FitToSquare(GrayImage source, int size = 64, int fit = 56)
    {
        var options = new NormalizerOptions { OutputSize = size, FitSize = fit, CenterOfMass = false };
        return new GrayImage(size, size, Normalize(source, options));
    }
}
