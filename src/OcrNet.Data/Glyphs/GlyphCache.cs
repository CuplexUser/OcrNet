using System.IO;
using System.IO.Compression;
using System.Text;
using OcrNet.Data.Imaging;

namespace OcrNet.Data.Glyphs;

/// <summary>
/// Persists rendered glyphs (<c>%LOCALAPPDATA%\OcrNet\glyphcache.bin</c>) so fonts are rasterized only
/// once. Entries are keyed by <see cref="FontFace.Key"/> and character.
/// </summary>
public sealed class GlyphCache
{
    private static readonly byte[] Magic = "OCRG"u8.ToArray();
    private readonly Dictionary<(string Face, char Character), GrayImage> _glyphs = [];
    private readonly HashSet<string> _renderedFaces = [];

    public static string DefaultPath =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OcrNet", "glyphcache.bin");

    public int Count => _glyphs.Count;

    public bool TryGet(FontFace face, char c, out GrayImage image) => _glyphs.TryGetValue((face.Key, c), out image!);

    /// <summary>
    /// Makes sure every face has been rendered, rendering only faces that are not cached yet.
    /// Characters a font lacks are simply absent from the cache.
    /// Returns true when something new was rendered (so the cache should be saved).
    /// </summary>
    public bool EnsureRendered(IReadOnlyList<FontFace> faces, string characters, int threads,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        var missing = faces.Where(f => !_renderedFaces.Contains(f.Key)).ToList();
        if (missing.Count == 0)
            return false;

        // Render the full printable ASCII range once per face so later character-set changes hit the cache.
        string all = new([.. Enumerable.Range(33, 94).Select(i => (char)i).Union(characters).Distinct()]);
        foreach (var (face, c, image) in GlyphRenderer.Render(missing, all, threads, progress, cancellationToken))
            _glyphs[(face.Key, c)] = image;
        foreach (var face in missing)
            _renderedFaces.Add(face.Key);
        return true;
    }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string temp = path + ".tmp";
        using (var file = File.Create(temp))
        {
            file.Write(Magic);
            using var brotli = new BrotliStream(file, CompressionLevel.Fastest);
            using var w = new BinaryWriter(brotli, Encoding.UTF8);
            w.Write(GlyphRenderer.Version);
            w.Write(GlyphRenderer.Size);
            w.Write(_renderedFaces.Count);
            foreach (var face in _renderedFaces)
                w.Write(face);
            w.Write(_glyphs.Count);
            foreach (var ((face, c), image) in _glyphs)
            {
                w.Write(face);
                w.Write(c);
                w.Write(image.ToBytes());
            }
        }
        File.Move(temp, path, overwrite: true);
    }

    /// <summary>Loads the cache, or returns an empty one when the file is missing, stale or damaged.</summary>
    public static GlyphCache Load(string path)
    {
        var cache = new GlyphCache();
        if (!File.Exists(path))
            return cache;
        try
        {
            using var file = File.OpenRead(path);
            Span<byte> magic = stackalloc byte[4];
            file.ReadExactly(magic);
            if (!magic.SequenceEqual(Magic))
                return cache;
            using var brotli = new BrotliStream(file, CompressionMode.Decompress);
            using var r = new BinaryReader(brotli, Encoding.UTF8);
            if (r.ReadInt32() != GlyphRenderer.Version || r.ReadInt32() != GlyphRenderer.Size)
                return cache;
            int faces = r.ReadInt32();
            for (int i = 0; i < faces; i++)
                cache._renderedFaces.Add(r.ReadString());
            int count = r.ReadInt32();
            int pixels = GlyphRenderer.Size * GlyphRenderer.Size;
            for (int i = 0; i < count; i++)
            {
                string face = r.ReadString();
                char c = r.ReadChar();
                cache._glyphs[(face, c)] = GrayImage.FromBytes(GlyphRenderer.Size, GlyphRenderer.Size, r.ReadBytes(pixels));
            }
            return cache;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or EndOfStreamException)
        {
            return new GlyphCache();
        }
    }
}
