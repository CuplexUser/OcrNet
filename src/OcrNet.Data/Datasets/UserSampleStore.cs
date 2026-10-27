using System.IO;
using System.Text;
using OcrNet.Data.Glyphs;
using OcrNet.Data.Imaging;

namespace OcrNet.Data.Datasets;

/// <summary>A drawing labelled by the user, stored as a 64×64 fitted image like rendered glyphs.</summary>
public sealed record UserSample(char Character, GrayImage Image, DateTime CreatedUtc);

/// <summary>
/// The user's own labelled drawings (<c>%LOCALAPPDATA%\OcrNet\user-samples.bin</c>). They are mixed
/// into training so the network adapts to the user's handwriting.
/// </summary>
public sealed class UserSampleStore
{
    private static readonly byte[] Magic = "OCRU"u8.ToArray();
    private const int FormatVersion = 1;
    private readonly List<UserSample> _samples = [];

    public static string DefaultPath =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OcrNet", "user-samples.bin");

    public IReadOnlyList<UserSample> Samples => _samples;

    public int Count => _samples.Count;

    public void Add(char character, GrayImage drawing) =>
        _samples.Add(new UserSample(character, Normalizer.FitToSquare(drawing, GlyphRenderer.Size, GlyphRenderer.Fit), DateTime.UtcNow));

    public void RemoveLast()
    {
        if (_samples.Count > 0)
            _samples.RemoveAt(_samples.Count - 1);
    }

    public void Clear() => _samples.Clear();

    public Dictionary<char, int> CountByCharacter() =>
        _samples.GroupBy(s => s.Character).ToDictionary(g => g.Key, g => g.Count());

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = File.Create(path);
        using var w = new BinaryWriter(file, Encoding.UTF8);
        w.Write(Magic);
        w.Write(FormatVersion);
        w.Write(_samples.Count);
        foreach (var s in _samples)
        {
            w.Write(s.Character);
            w.Write(s.CreatedUtc.Ticks);
            w.Write(s.Image.Width);
            w.Write(s.Image.Height);
            w.Write(s.Image.ToBytes());
        }
    }

    public static UserSampleStore Load(string path)
    {
        var store = new UserSampleStore();
        if (!File.Exists(path))
            return store;
        try
        {
            using var file = File.OpenRead(path);
            using var r = new BinaryReader(file, Encoding.UTF8);
            if (!r.ReadBytes(4).AsSpan().SequenceEqual(Magic) || r.ReadInt32() != FormatVersion)
                return store;
            int count = r.ReadInt32();
            for (int i = 0; i < count; i++)
            {
                char c = r.ReadChar();
                var created = new DateTime(r.ReadInt64(), DateTimeKind.Utc);
                int width = r.ReadInt32(), height = r.ReadInt32();
                var image = GrayImage.FromBytes(width, height, r.ReadBytes(width * height));
                store._samples.Add(new UserSample(c, image, created));
            }
        }
        catch (Exception ex) when (ex is IOException or EndOfStreamException)
        {
            // A damaged file loses its samples rather than crashing the app.
        }
        return store;
    }
}
