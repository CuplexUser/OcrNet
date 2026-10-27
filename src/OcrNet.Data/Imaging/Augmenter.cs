using OcrNet.Core.Compute;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

namespace OcrNet.Data.Imaging;

/// <summary>Random distortions that make clean font glyphs look more like handwriting.</summary>
public sealed class AugmentationOptions
{
    public bool Enabled { get; set; } = true;

    public float MaxRotationDegrees { get; set; } = 12f;

    /// <summary>Maximum horizontal shear (tan of the slant angle).</summary>
    public float MaxShear { get; set; } = 0.3f;

    public float MinScale { get; set; } = 0.85f;

    public float MaxScale { get; set; } = 1.1f;

    /// <summary>Maximum relative change of the width/height ratio.</summary>
    public float MaxAspectChange { get; set; } = 0.2f;

    /// <summary>Probability of an elastic (wobbly-line) distortion.</summary>
    public float ElasticProbability { get; set; } = 0.6f;

    /// <summary>Maximum elastic displacement in source pixels (64×64 glyph space).</summary>
    public float ElasticStrength { get; set; } = 2.5f;

    /// <summary>Probability of thickening or thinning the strokes.</summary>
    public float StrokeProbability { get; set; } = 0.6f;

    /// <summary>Standard deviation of Gaussian pixel noise added to the network input.</summary>
    public float Noise { get; set; } = 0.02f;

    public AugmentationOptions Clone() => (AugmentationOptions)MemberwiseClone();
}

/// <summary>
/// Applies a random affine transform (rotation, shear, scale, aspect), an optional elastic
/// distortion and stroke-width change to a 64×64 source glyph. Thread-safe as long as each thread
/// passes its own <see cref="FastRandom"/> and destination image.
/// </summary>
public static class Augmenter
{
    private const int Grid = 5; // control points per side for the elastic displacement field

    [ThreadStatic]
    private static float[]? _morphScratch;

    /// <summary>Returns a distorted copy of <paramref name="source"/>.</summary>
    public static GrayImage Distort(GrayImage source, AugmentationOptions o, FastRandom rng)
    {
        var result = new GrayImage(source.Width, source.Height);
        Distort(source, o, rng, result);
        return result;
    }

    /// <summary>Writes a distorted copy of <paramref name="source"/> into <paramref name="destination"/> (same size).</summary>
    public static void Distort(GrayImage source, AugmentationOptions o, FastRandom rng, GrayImage destination)
    {
        int w = source.Width, h = source.Height;
        if (!o.Enabled)
        {
            source.Pixels.CopyTo(destination.Pixels, 0);
            return;
        }

        float angle = rng.NextFloat(-o.MaxRotationDegrees, o.MaxRotationDegrees) * MathF.PI / 180f;
        float shear = rng.NextFloat(-o.MaxShear, o.MaxShear);
        float scale = rng.NextFloat(o.MinScale, o.MaxScale);
        float aspect = 1f + rng.NextFloat(-o.MaxAspectChange, o.MaxAspectChange);
        float sx = scale * aspect, sy = scale / aspect;

        // Forward transform A = R · Shear · S about the image center; pixels are sampled with A⁻¹.
        float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
        float a00 = cos * sx, a01 = (cos * shear - sin) * sy;
        float a10 = sin * sx, a11 = (sin * shear + cos) * sy;
        float det = a00 * a11 - a01 * a10;
        float i00 = a11 / det, i01 = -a01 / det, i10 = -a10 / det, i11 = a00 / det;
        float cx = (w - 1) / 2f, cy = (h - 1) / 2f;

        Span<float> gx = stackalloc float[Grid * Grid];
        Span<float> gy = stackalloc float[Grid * Grid];
        bool elastic = rng.NextFloat() < o.ElasticProbability && o.ElasticStrength > 0;
        float maxDisplacement = 0f;
        if (elastic)
        {
            float sigma = o.ElasticStrength * 0.6f;
            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = Math.Clamp(rng.NextGaussian(), -3f, 3f) * sigma;
                gy[i] = Math.Clamp(rng.NextGaussian(), -3f, 3f) * sigma;
            }
            maxDisplacement = 3f * sigma;
        }

        // Only the area the transformed ink can reach needs sampling; the rest stays background.
        destination.Pixels.AsSpan().Clear();
        var ink = source.InkBounds(0f);
        if (ink is null)
            return;
        var (bx, by, bw, bh) = ink.Value;
        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
        foreach (var (px, py) in new[] { (bx - 1f, by - 1f), (bx + bw + 0f, by - 1f), (bx - 1f, by + bh + 0f), (bx + bw + 0f, by + bh + 0f) })
        {
            float ox = a00 * (px - cx) + a01 * (py - cy) + cx;
            float oy = a10 * (px - cx) + a11 * (py - cy) + cy;
            minX = MathF.Min(minX, ox); maxX = MathF.Max(maxX, ox);
            minY = MathF.Min(minY, oy); maxY = MathF.Max(maxY, oy);
        }
        // Elastic offsets are in source space; scale them into output space conservatively.
        float margin = maxDisplacement * MathF.Max(sx, 1f / sy) * 1.5f + 2f;
        int x0 = Math.Max(0, (int)(minX - margin)), x1 = Math.Min(w - 1, (int)(maxX + margin));
        int y0 = Math.Max(0, (int)(minY - margin)), y1 = Math.Min(h - 1, (int)(maxY + margin));

        float[] dst = destination.Pixels;
        float gridScaleX = (Grid - 1f) / (w - 1), gridScaleY = (Grid - 1f) / (h - 1);
        Span<float> rowDx = stackalloc float[Grid];
        Span<float> rowDy = stackalloc float[Grid];
        for (int y = y0; y <= y1; y++)
        {
            float py = y - cy;
            float rowX = i01 * py + cx, rowY = i11 * py + cy;
            int row = y * w;
            if (elastic)
            {
                // The field is bilinear in the grid: interpolate the grid rows once, then only along x.
                InterpolateRow(gx, y * gridScaleY, rowDx);
                InterpolateRow(gy, y * gridScaleY, rowDy);
            }
            if (!elastic)
            {
                for (int x = x0; x <= x1; x++)
                {
                    float px = x - cx;
                    dst[row + x] = source.Sample(i00 * px + rowX, i10 * px + rowY);
                }
                continue;
            }

            // Along a row both the affine part and the elastic offset are linear between grid
            // columns, so the source position advances by a constant step within each segment.
            for (int k = 0; k < Grid - 1; k++)
            {
                int xs = Math.Max(x0, (int)MathF.Ceiling(k / gridScaleX));
                int xe = Math.Min(x1, k == Grid - 2 ? w - 1 : (int)MathF.Ceiling((k + 1) / gridScaleX) - 1);
                if (xs > xe)
                    continue;
                float slopeDx = (rowDx[k + 1] - rowDx[k]) * gridScaleX;
                float slopeDy = (rowDy[k + 1] - rowDy[k]) * gridScaleX;
                float f = xs * gridScaleX - k;
                float px = xs - cx;
                float srcX = i00 * px + rowX + rowDx[k] + (rowDx[k + 1] - rowDx[k]) * f;
                float srcY = i10 * px + rowY + rowDy[k] + (rowDy[k + 1] - rowDy[k]) * f;
                float stepX = i00 + slopeDx, stepY = i10 + slopeDy;
                for (int x = xs; x <= xe; x++)
                {
                    dst[row + x] = source.Sample(srcX, srcY);
                    srcX += stepX;
                    srcY += stepY;
                }
            }
        }

        if (rng.NextFloat() < o.StrokeProbability)
        {
            bool thicken = rng.NextFloat() < 0.6f;
            int passes = rng.NextFloat() < 0.3f ? 2 : 1;
            for (int p = 0; p < passes; p++)
                Morph(destination, thicken);
        }
    }

    /// <summary>Adds clamped Gaussian noise to a normalized network input.</summary>
    public static void AddNoise(Span<float> input, float stdDev, FastRandom rng)
    {
        if (stdDev <= 0)
            return;
        // Triangular noise (sum of two uniforms) has the requested standard deviation and is much
        // cheaper than true Gaussian samples.
        float scale = stdDev * MathF.Sqrt(6f);
        for (int i = 0; i < input.Length; i++)
            input[i] = Math.Clamp(input[i] + (rng.NextFloat() + rng.NextFloat() - 1f) * scale, 0f, 1f);
    }

    /// <summary>Linearly interpolates between the two grid rows around <paramref name="v"/>.</summary>
    private static void InterpolateRow(ReadOnlySpan<float> grid, float v, Span<float> row)
    {
        int y0 = Math.Min((int)v, Grid - 2);
        float f = v - y0;
        for (int k = 0; k < Grid; k++)
        {
            float a = grid[y0 * Grid + k], b = grid[(y0 + 1) * Grid + k];
            row[k] = a + (b - a) * f;
        }
    }

    /// <summary>
    /// In-place separable 3×3 grayscale dilation (thicken) or erosion (thin); outside the image counts
    /// as background. Erosion is undone when it would remove most of the ink (very thin fonts).
    /// </summary>
    private static void Morph(GrayImage image, bool dilate)
    {
        int w = image.Width, h = image.Height, n = w * h;
        int pw = w + 2, ph = h + 2;      // zero border of one pixel on every side
        float[] p = image.Pixels;
        int needed = pw * ph * 2;
        var scratch = _morphScratch is { } s && s.Length >= needed ? s : _morphScratch = new float[needed];
        var padded = scratch.AsSpan(0, pw * ph);
        var horizontal = scratch.AsSpan(pw * ph, pw * ph);
        padded.Clear();
        for (int y = 0; y < h; y++)
            p.AsSpan(y * w, w).CopyTo(padded.Slice((y + 1) * pw + 1, w));

        // Horizontal pass over the padded rows (output column x+1 uses inputs x, x+1, x+2).
        horizontal.Clear();
        for (int y = 1; y <= h; y++)
        {
            var src = padded.Slice(y * pw, pw);
            var dst = horizontal.Slice(y * pw + 1, w);
            MinMax3(src[..w], src.Slice(1, w), src.Slice(2, w), dst, dilate);
        }

        // Vertical pass written straight back into the image.
        float before = 0f, after = 0f;
        for (int y = 0; y < h; y++)
        {
            var dst = p.AsSpan(y * w, w);
            if (!dilate)
                before += TensorPrimitives.Sum<float>(dst);
            MinMax3(horizontal.Slice(y * pw + 1, w), horizontal.Slice((y + 1) * pw + 1, w),
                horizontal.Slice((y + 2) * pw + 1, w), dst, dilate);
            if (!dilate)
                after += TensorPrimitives.Sum<float>(dst);
        }

        if (!dilate && after < before * 0.35f)
        {
            for (int y = 0; y < h; y++)
                padded.Slice((y + 1) * pw + 1, w).CopyTo(p.AsSpan(y * w, w));
        }
    }

    private static void MinMax3(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, Span<float> dst, bool max)
    {
        if (max)
        {
            TensorPrimitives.Max(a, b, dst);
            TensorPrimitives.Max(dst, c, dst);
        }
        else
        {
            TensorPrimitives.Min(a, b, dst);
            TensorPrimitives.Min(dst, c, dst);
        }
    }
}
