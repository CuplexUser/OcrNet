using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using OcrNet.Data.Imaging;

namespace OcrNet.Data.Datasets;

/// <summary>
/// MNIST-style handwritten digits, kept as bytes (60,000 digits take 47 MB rather than 188 MB as
/// float images). <see cref="Image"/> converts one on demand.
/// </summary>
public sealed class MnistSet(int width, int height, byte[] pixels, byte[] labels, string imagesPath)
{
    public int Width { get; } = width;

    public int Height { get; } = height;

    public int Count => Labels.Length;

    /// <summary>The digit (0–9) of each image.</summary>
    public byte[] Labels { get; } = labels;

    /// <summary>The file the images were read from.</summary>
    public string ImagesPath { get; } = imagesPath;

    public GrayImage Image(int index) => GrayImage.FromBytes(Width, Height, pixels.AsSpan(index * Width * Height, Width * Height));
}

/// <summary>
/// Reads MNIST-style IDX files (optionally gzipped) so a digit model can be trained on, or evaluated
/// on, real handwriting. The files are not shipped with OcrNet.
/// </summary>
public static class MnistReader
{
    private const int ImagesMagic = 0x00000803;
    private const int LabelsMagic = 0x00000801;

    /// <summary>
    /// The labels file that belongs to <paramref name="imagesPath"/> by MNIST naming ("images" →
    /// "labels", "idx3" → "idx1", gzipped or not, e.g. <c>t10k-images-idx3-ubyte</c> →
    /// <c>t10k-labels-idx1-ubyte.gz</c>), or null if there is none. Looks next to the images and in
    /// the matching sibling folder, since some unzip tools put each file in a folder of its own name
    /// (<c>t10k-images-idx3-ubyte\t10k-images.idx3-ubyte</c>).
    /// </summary>
    public static string? FindLabelsFile(string imagesPath) =>
        FindCompanion(imagesPath, name => name.Replace("images", "labels", StringComparison.OrdinalIgnoreCase)
            .Replace("idx3", "idx1", StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// The test images that belong to the training images at <paramref name="trainImagesPath"/>
    /// ("train" → "t10k"), looked for the same way as <see cref="FindLabelsFile"/>, or null.
    /// </summary>
    public static string? FindTestImagesFile(string trainImagesPath) =>
        FindCompanion(trainImagesPath, name => name.Replace("train", "t10k", StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// The file named <paramref name="rename"/>(name of <paramref name="path"/>), gzipped or not,
    /// next to <paramref name="path"/> or in the sibling folder renamed the same way.
    /// </summary>
    private static string? FindCompanion(string path, Func<string, string> rename)
    {
        string name = Path.GetFileName(path);
        string companion = rename(name);
        if (companion.Equals(name, StringComparison.OrdinalIgnoreCase))
            return null;
        if (companion.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
            companion = companion[..^3];

        string folder = Path.GetDirectoryName(Path.GetFullPath(path))!;
        var folders = new List<string> { folder };
        if (Path.GetDirectoryName(folder) is { } parent && rename(Path.GetFileName(folder)) is var sibling
            && !sibling.Equals(Path.GetFileName(folder), StringComparison.OrdinalIgnoreCase))
            folders.Add(Path.Combine(parent, sibling));
        return folders.SelectMany(f => new[] { companion, companion + ".gz" }.Select(n => Path.Combine(f, n))).FirstOrDefault(File.Exists);
    }

    /// <summary>Reads an images file and the labels file found for it by <see cref="FindLabelsFile"/>.</summary>
    public static MnistSet Read(string imagesPath) =>
        Read(imagesPath, FindLabelsFile(imagesPath)
            ?? throw new FileNotFoundException($"No labels file was found for {Path.GetFileName(imagesPath)}."));

    public static MnistSet Read(string imagesPath, string labelsPath, int maxCount = int.MaxValue)
    {
        using var imageStream = Open(imagesPath);
        using var labelStream = Open(labelsPath);

        // The first 8 bytes are common to both kinds, so a swapped labels file is recognized before
        // reading the rest of the image header.
        Span<byte> header = stackalloc byte[16];
        imageStream.ReadExactly(header[..8]);
        int magic = BinaryPrimitives.ReadInt32BigEndian(header);
        if (magic == LabelsMagic)
            throw new InvalidDataException($"{Path.GetFileName(imagesPath)} is an IDX labels file; choose the images file (…images-idx3-ubyte) instead.");
        if (magic != ImagesMagic)
            throw new InvalidDataException($"{Path.GetFileName(imagesPath)} is not an IDX image file.");
        imageStream.ReadExactly(header[8..]);
        int count = BinaryPrimitives.ReadInt32BigEndian(header[4..]);
        int rows = BinaryPrimitives.ReadInt32BigEndian(header[8..]);
        int cols = BinaryPrimitives.ReadInt32BigEndian(header[12..]);

        labelStream.ReadExactly(header[..8]);
        if (BinaryPrimitives.ReadInt32BigEndian(header) != LabelsMagic)
            throw new InvalidDataException($"{Path.GetFileName(labelsPath)} is not an IDX labels file.");
        if (BinaryPrimitives.ReadInt32BigEndian(header[4..]) != count)
            throw new InvalidDataException("Image and label files contain different numbers of items.");

        count = Math.Min(count, maxCount);
        var pixels = new byte[count * rows * cols];
        var labels = new byte[count];
        imageStream.ReadExactly(pixels);
        labelStream.ReadExactly(labels);
        if (labels.Any(l => l > 9))
            throw new InvalidDataException($"{Path.GetFileName(labelsPath)} contains labels other than the digits 0–9.");
        return new MnistSet(cols, rows, pixels, labels, imagesPath);
    }

    /// <summary>
    /// Opens a file, decompressing it if it is gzipped. Decided by content rather than the ".gz"
    /// extension, since browsers sometimes unpack downloads without renaming them, or the reverse.
    /// </summary>
    private static Stream Open(string path)
    {
        Stream stream = File.OpenRead(path);
        bool gzipped = stream.ReadByte() == 0x1F && stream.ReadByte() == 0x8B;
        stream.Position = 0;
        return gzipped
            ? new BufferedStream(new GZipStream(stream, CompressionMode.Decompress))
            : new BufferedStream(stream);
    }
}
