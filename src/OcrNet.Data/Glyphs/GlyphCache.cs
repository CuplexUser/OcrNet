using System.IO;
using System.IO.Compression;
using System.Text;
using OcrNet.Data.Datasets;
using OcrNet.Data.Imaging;

namespace OcrNet.Data.Glyphs;

/// <summary>
/// Persists rendered glyphs (<c>%LOCALAPPDATA%\OcrNet\glyphcache.bin</c>) so fonts are rasterized only
/// once. Entries are keyed by <see cref="FontFace.Key"/> and character. The cache also remembers
/// which characters were attempted per face, so characters added later (a custom set, say) are
/// rendered on demand without redoing the rest.
/// </summary>
public sealed class GlyphCache
{
    private static readonly byte[] Magic = "OCRG"u8.ToArray();

    /// <summary>Bumped when the file layout changes; older files are discarded and re-rendered.</summary>
    private const int FormatVersion = 2;

    /// <summary>Rendered for every face up front: printable ASCII, the Swedish letters and the symbol set (£, €).</summary>
    public static readonly string BaseCharacters = new([..
        Enumerable.Range(33, 94).Select(i => (char)i)
            .Concat(CharacterSet.SwedishUpper + CharacterSet.SwedishLower + CharacterSet.Symbols)
            .Distinct()]);

    private readonly Dictionary<(string Face, char Character), GrayImage> _glyphs = [];

    /// <summary>Characters already attempted per face (whether or not the font has them).</summary>
    private readonly Dictionary<string, HashSet<char>> _attempted = [];

    public static string DefaultPath =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OcrNet", "glyphcache.bin");

    public int Count => _glyphs.Count;

    public bool TryGet(FontFace face, char c, out GrayImage image) => _glyphs.TryGetValue((face.Key, c), out image!);

    /// <summary>
    /// Makes sure every face has been rendered with <see cref="BaseCharacters"/> and
    /// <paramref name="characters"/>, rendering only what is not cached yet. Characters a font lacks
    /// are simply absent from the cache. Returns true when something new was rendered (so the cache
    /// should be saved).
    /// </summary>
    public bool EnsureRendered(IReadOnlyList<FontFace> faces, string characters, int threads,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        var wanted = BaseCharacters.Union(characters.Where(c => !char.IsWhiteSpace(c))).ToArray();
        // Faces usually miss the same characters, so render them in groups that share one character list.
        var groups = faces
            .Select(face => (face, missing: new string([.. wanted.Where(c => !_attempted.TryGetValue(face.Key, out var done) || !done.Contains(c))])))
            .Where(x => x.missing.Length > 0)
            .GroupBy(x => x.missing, x => x.face)
            .ToList();
        if (groups.Count == 0)
            return false;

        int total = groups.Sum(g => g.Count() * g.Key.Length), finished = 0;
        foreach (var group in groups)
        {
            int weight = group.Count() * group.Key.Length;
            var groupProgress = progress is null ? null : new Progress<double>(f => progress.Report((finished + f * weight) / total));
            foreach (var (face, c, image) in GlyphRenderer.Render([.. group], group.Key, threads, groupProgress, cancellationToken))
                _glyphs[(face.Key, c)] = image;
            foreach (var face in group)
            {
                if (!_attempted.TryGetValue(face.Key, out var done))
                    _attempted[face.Key] = done = [];
                done.UnionWith(group.Key);
            }
            finished += weight;
        }
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
            w.Write(FormatVersion);
            w.Write(GlyphRenderer.Version);
            w.Write(GlyphRenderer.Size);
            w.Write(_attempted.Count);
            foreach (var (face, characters) in _attempted)
            {
                w.Write(face);
                w.Write(new string([.. characters]));
            }
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
            if (r.ReadInt32() != FormatVersion || r.ReadInt32() != GlyphRenderer.Version || r.ReadInt32() != GlyphRenderer.Size)
                return cache;
            int faces = r.ReadInt32();
            for (int i = 0; i < faces; i++)
                cache._attempted[r.ReadString()] = [.. r.ReadString()];
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
