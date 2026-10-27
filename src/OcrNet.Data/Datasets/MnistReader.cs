using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using OcrNet.Data.Imaging;

namespace OcrNet.Data.Datasets;

/// <summary>
/// Reads MNIST-style IDX files (optionally gzipped) so a font-trained digit model can be evaluated
/// on real handwriting. The files are not shipped with OcrNet.
/// </summary>
public static class MnistReader
{
    public static (List<GrayImage> Images, List<byte> Labels) Read(string imagesPath, string labelsPath, int maxCount = int.MaxValue)
    {
        using var imageStream = Open(imagesPath);
        using var labelStream = Open(labelsPath);

        Span<byte> header = stackalloc byte[16];
        imageStream.ReadExactly(header);
        if (BinaryPrimitives.ReadInt32BigEndian(header) != 0x00000803)
            throw new InvalidDataException("Not an IDX image file.");
        int count = BinaryPrimitives.ReadInt32BigEndian(header[4..]);
        int rows = BinaryPrimitives.ReadInt32BigEndian(header[8..]);
        int cols = BinaryPrimitives.ReadInt32BigEndian(header[12..]);

        labelStream.ReadExactly(header[..8]);
        if (BinaryPrimitives.ReadInt32BigEndian(header) != 0x00000801)
            throw new InvalidDataException("Not an IDX label file.");
        if (BinaryPrimitives.ReadInt32BigEndian(header[4..]) != count)
            throw new InvalidDataException("Image and label files contain different numbers of items.");

        count = Math.Min(count, maxCount);
        var images = new List<GrayImage>(count);
        var labels = new List<byte>(count);
        var buffer = new byte[rows * cols];
        for (int i = 0; i < count; i++)
        {
            imageStream.ReadExactly(buffer);
            images.Add(GrayImage.FromBytes(cols, rows, buffer));
            labels.Add((byte)labelStream.ReadByte());
        }
        return (images, labels);
    }

    private static Stream Open(string path)
    {
        Stream stream = File.OpenRead(path);
        return path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase)
            ? new BufferedStream(new GZipStream(stream, CompressionMode.Decompress))
            : new BufferedStream(stream);
    }
}
