using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OcrNet.Data.Imaging;

namespace OcrNet.Data.Glyphs;

/// <summary>
/// Rasterizes glyph outlines of installed fonts into 64×64 images (ink fitted to 56 px, centered).
/// Outlines come straight from the font file, so missing glyphs are never replaced by fallback fonts.
/// WPF rendering needs STA threads, so the work runs on dedicated worker threads.
/// </summary>
public static class GlyphRenderer
{
    public const int Size = 64;
    public const int Fit = 56;

    /// <summary>Bumped whenever rendering changes, which invalidates the on-disk cache.</summary>
    public const int Version = 1;

    /// <summary>
    /// Renders every character of <paramref name="characters"/> for every face. Missing glyphs are
    /// skipped. <paramref name="progress"/> receives the fraction done.
    /// </summary>
    public static List<(FontFace Face, char Character, GrayImage Image)> Render(
        IReadOnlyList<FontFace> faces, string characters, int threads, IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<(FontFace, char, GrayImage)>();
        if (faces.Count == 0)
            return results;

        int workerCount = Math.Clamp(threads, 1, faces.Count);
        int next = -1, done = 0;
        var gate = new object();
        Exception? failure = null;

        var workers = Enumerable.Range(0, workerCount).Select(_ =>
        {
            var thread = new Thread(() =>
            {
                try
                {
                    int i;
                    while ((i = Interlocked.Increment(ref next)) < faces.Count && !cancellationToken.IsCancellationRequested)
                    {
                        var rendered = RenderFace(faces[i], characters);
                        lock (gate)
                            results.AddRange(rendered.Select(r => (faces[i], r.Character, r.Image)));
                        progress?.Report((double)Interlocked.Increment(ref done) / faces.Count);
                    }
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
            })
            {
                IsBackground = true,
                Name = "Glyph renderer",
            };
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();
            return thread;
        }).ToList();

        foreach (var worker in workers)
            worker.Join();
        cancellationToken.ThrowIfCancellationRequested();
        if (failure is not null)
            throw new InvalidOperationException("Glyph rendering failed: " + failure.Message, failure);
        return results;
    }

    /// <summary>Renders one face on the calling thread, which must be STA.</summary>
    public static List<(char Character, GrayImage Image)> RenderFace(FontFace face, string characters)
    {
        var results = new List<(char, GrayImage)>();
        var typeface = FontCatalog.CreateTypeface(new FontFamily(face.Family), face.Variant);
        if (!typeface.TryGetGlyphTypeface(out var glyphs))
            return results;

        var bitmap = new RenderTargetBitmap(Size, Size, 96, 96, PixelFormats.Pbgra32);
        var pixels = new byte[Size * Size * 4];
        foreach (char c in characters)
        {
            if (!glyphs.CharacterToGlyphMap.TryGetValue(c, out ushort index))
                continue;
            var image = RenderGlyph(glyphs, index, bitmap, pixels);
            if (image is not null)
                results.Add((c, image));
        }
        return results;
    }

    private static GrayImage? RenderGlyph(GlyphTypeface glyphs, ushort index, RenderTargetBitmap bitmap, byte[] pixels)
    {
        const double em = 200;
        Geometry outline = glyphs.GetGlyphOutline(index, em, em);
        bool bold = glyphs.StyleSimulations.HasFlag(StyleSimulations.BoldSimulation);
        bool italic = glyphs.StyleSimulations.HasFlag(StyleSimulations.ItalicSimulation);
        if (italic)
            outline = Transform(outline, new Matrix(1, 0, -0.33, 1, 0, 0));
        var pen = bold ? new Pen(Brushes.White, em * 0.035) { LineJoin = PenLineJoin.Round } : null;

        Rect bounds = pen is null ? outline.Bounds : outline.GetRenderBounds(pen);
        if (bounds.IsEmpty || bounds.Width < 1e-3 || bounds.Height < 1e-3)
            return null;

        double scale = Fit / Math.Max(bounds.Width, bounds.Height);
        double offsetX = (Size - bounds.Width * scale) / 2 - bounds.X * scale;
        double offsetY = (Size - bounds.Height * scale) / 2 - bounds.Y * scale;

        var visual = new DrawingVisual();
        using (var dc = visual.RenderOpen())
        {
            dc.DrawRectangle(Brushes.Black, null, new Rect(0, 0, Size, Size));
            dc.PushTransform(new MatrixTransform(scale, 0, 0, scale, offsetX, offsetY));
            dc.DrawGeometry(Brushes.White, pen, outline);
            dc.Pop();
        }
        bitmap.Clear();
        bitmap.Render(visual);
        bitmap.CopyPixels(pixels, Size * 4, 0);

        var image = new GrayImage(Size, Size);
        for (int i = 0; i < image.Pixels.Length; i++)
            image.Pixels[i] = pixels[i * 4 + 2] / 255f; // red channel of white-on-black
        return image;
    }

    private static Geometry Transform(Geometry geometry, Matrix matrix)
    {
        var clone = geometry.Clone();
        clone.Transform = new MatrixTransform(matrix);
        return clone;
    }
}
