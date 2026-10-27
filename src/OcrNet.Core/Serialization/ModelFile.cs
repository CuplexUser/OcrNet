using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using OcrNet.Core.Layers;
using OcrNet.Core.Network;
using OcrNet.Core.Optimizers;
using OcrNet.Core.Tensors;
using OcrNet.Core.Training;

namespace OcrNet.Core.Serialization;

/// <summary>A loaded model plus the optimizer state needed to continue training, if it was saved.</summary>
public sealed record ModelFileContents(NeuralNetwork Network, Optimizer? Optimizer);

/// <summary>
/// Reads and writes the binary <c>.ocrnet</c> model format (little-endian):
/// <code>
/// "OCRN" | u16 version | u16 flags | i64 bodyLength | body (Brotli when flag bit 1) | u32 CRC-32 of all preceding bytes
/// </code>
/// The body holds the metadata, input shape, class labels, each layer (kind, config, named arrays)
/// and optionally the optimizer state. See README.md for the full layout.
/// </summary>
public static class ModelFile
{
    public const string Extension = ".ocrnet";
    public const ushort FormatVersion = 1;

    private static readonly byte[] Magic = "OCRN"u8.ToArray();
    private const ushort FlagOptimizerState = 1 << 0;
    private const ushort FlagBrotli = 1 << 1;
    private const int HeaderSize = 4 + 2 + 2 + 8;

    public static void Save(string path, NeuralNetwork network, Optimizer? optimizer = null, bool compress = true)
    {
        string temp = path + ".tmp";
        using (var stream = File.Create(temp))
            Save(stream, network, optimizer, compress);
        File.Move(temp, path, overwrite: true);
    }

    public static void Save(Stream output, NeuralNetwork network, Optimizer? optimizer = null, bool compress = true)
    {
        byte[] body;
        lock (network.SyncRoot)
            body = WriteBody(network, optimizer);

        if (compress)
        {
            using var compressed = new MemoryStream();
            using (var brotli = new BrotliStream(compressed, CompressionLevel.Fastest, leaveOpen: true))
                brotli.Write(body);
            body = compressed.ToArray();
        }

        ushort flags = (ushort)((optimizer is not null ? FlagOptimizerState : 0) | (compress ? FlagBrotli : 0));
        var file = new byte[HeaderSize + body.Length + 4];
        Magic.CopyTo(file, 0);
        BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan(4), FormatVersion);
        BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan(6), flags);
        BinaryPrimitives.WriteInt64LittleEndian(file.AsSpan(8), body.Length);
        body.CopyTo(file, HeaderSize);
        uint crc = Crc32.Compute(file.AsSpan(0, HeaderSize + body.Length));
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(HeaderSize + body.Length), crc);
        output.Write(file);
    }

    public static ModelFileContents Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    public static ModelFileContents Load(Stream input)
    {
        using var buffer = new MemoryStream();
        input.CopyTo(buffer);
        var file = buffer.GetBuffer().AsSpan(0, (int)buffer.Length);

        if (file.Length < HeaderSize + 4 || !file[..4].SequenceEqual(Magic))
            throw new InvalidDataException("Not an OcrNet model file.");
        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(file[4..]);
        if (version > FormatVersion)
            throw new InvalidDataException($"Model format version {version} is newer than supported version {FormatVersion}.");
        ushort flags = BinaryPrimitives.ReadUInt16LittleEndian(file[6..]);
        long bodyLength = BinaryPrimitives.ReadInt64LittleEndian(file[8..]);
        if (bodyLength < 0 || HeaderSize + bodyLength + 4 != file.Length)
            throw new InvalidDataException("Model file is truncated or has an invalid length.");

        uint storedCrc = BinaryPrimitives.ReadUInt32LittleEndian(file[(HeaderSize + (int)bodyLength)..]);
        uint actualCrc = Crc32.Compute(file[..(HeaderSize + (int)bodyLength)]);
        if (storedCrc != actualCrc)
            throw new InvalidDataException("Model file is corrupted (CRC mismatch).");

        Stream body = new MemoryStream(file.Slice(HeaderSize, (int)bodyLength).ToArray());
        if ((flags & FlagBrotli) != 0)
            body = new BrotliStream(body, CompressionMode.Decompress);
        using var reader = new BinaryReader(body, Encoding.UTF8);
        return ReadBody(reader, (flags & FlagOptimizerState) != 0);
    }

    /// <summary>Deep copy via an in-memory save/load round trip.</summary>
    public static NeuralNetwork Clone(NeuralNetwork network)
    {
        using var stream = new MemoryStream();
        Save(stream, network, compress: false);
        stream.Position = 0;
        return Load(stream).Network;
    }

    private static byte[] WriteBody(NeuralNetwork network, Optimizer? optimizer)
    {
        using var stream = new MemoryStream();
        using var w = new BinaryWriter(stream, Encoding.UTF8);

        var meta = network.Metadata;
        w.Write(meta.Name);
        w.Write(meta.Description);
        w.Write(meta.CreatedUtc.Ticks);
        w.Write(meta.AppVersion);
        w.WriteDictionary(meta.Training);
        w.WriteDictionary(meta.Preprocessing);
        w.Write(meta.Metrics.Count);
        foreach (var (key, value) in meta.Metrics)
        {
            w.Write(key);
            w.Write(value);
        }
        w.Write(meta.History.Count);
        foreach (var h in meta.History)
        {
            w.Write(h.Epoch);
            w.Write(h.TrainLoss);
            w.Write(h.TrainAccuracy);
            w.Write(h.ValidationLoss);
            w.Write(h.ValidationAccuracy);
            w.Write(h.LearningRate);
            w.Write(h.Seconds);
        }

        w.Write(network.InputShape.C);
        w.Write(network.InputShape.H);
        w.Write(network.InputShape.W);
        w.Write(network.Labels.Count);
        foreach (var label in network.Labels)
            w.Write(label);

        w.Write(network.Layers.Count);
        foreach (var layer in network.Layers)
        {
            w.Write((ushort)layer.Kind);
            using (var configStream = new MemoryStream())
            {
                using (var cw = new BinaryWriter(configStream, Encoding.UTF8, leaveOpen: true))
                    layer.WriteConfig(cw);
                w.Write((int)configStream.Length);
                w.Write(configStream.GetBuffer(), 0, (int)configStream.Length);
            }
            var arrays = layer.Parameters.Cast<NamedArray>().Concat(layer.State).ToList();
            w.Write(arrays.Count);
            foreach (var array in arrays)
            {
                w.Write(array.Name);
                w.Write(array.Dims.Length);
                foreach (int d in array.Dims)
                    w.Write(d);
                w.WriteFloats(array.Value);
            }
        }

        w.Write(optimizer is not null);
        if (optimizer is not null)
        {
            w.Write((byte)optimizer.Kind);
            optimizer.WriteState(w);
        }

        w.Flush();
        return stream.ToArray();
    }

    private static ModelFileContents ReadBody(BinaryReader r, bool expectOptimizer)
    {
        var meta = new ModelMetadata
        {
            Name = r.ReadString(),
            Description = r.ReadString(),
            CreatedUtc = new DateTime(r.ReadInt64(), DateTimeKind.Utc),
            AppVersion = r.ReadString(),
            Training = r.ReadStringDictionary(),
            Preprocessing = r.ReadStringDictionary(),
        };
        int metricCount = r.ReadInt32();
        for (int i = 0; i < metricCount; i++)
            meta.Metrics[r.ReadString()] = r.ReadDouble();
        int historyCount = r.ReadInt32();
        for (int i = 0; i < historyCount; i++)
        {
            meta.History.Add(new EpochRecord(r.ReadInt32(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle(),
                r.ReadSingle(), r.ReadSingle(), r.ReadDouble()));
        }

        var inputShape = new TensorShape(r.ReadInt32(), r.ReadInt32(), r.ReadInt32());
        int labelCount = r.ReadInt32();
        var labels = new string[labelCount];
        for (int i = 0; i < labelCount; i++)
            labels[i] = r.ReadString();

        int layerCount = r.ReadInt32();
        var layers = new List<Layer>(layerCount);
        var shape = inputShape;
        for (int i = 0; i < layerCount; i++)
        {
            var kind = (LayerKind)r.ReadUInt16();
            int configLength = r.ReadInt32();
            var config = r.ReadBytes(configLength);
            Layer layer;
            using (var cr = new BinaryReader(new MemoryStream(config), Encoding.UTF8))
                layer = LayerFactory.Create(kind, shape, cr);

            var arrays = layer.Parameters.Cast<NamedArray>().Concat(layer.State).ToDictionary(a => a.Name);
            int arrayCount = r.ReadInt32();
            for (int a = 0; a < arrayCount; a++)
            {
                string name = r.ReadString();
                int rank = r.ReadInt32();
                var dims = new int[rank];
                for (int d = 0; d < rank; d++)
                    dims[d] = r.ReadInt32();
                float[] values = r.ReadFloats();
                if (!arrays.TryGetValue(name, out var target) || target.Length != values.Length)
                    throw new InvalidDataException($"Layer {i} ({kind}) has an unexpected array '{name}'.");
                values.CopyTo(target.Value, 0);
            }
            layers.Add(layer);
            shape = layer.OutputShape;
        }

        var network = new NeuralNetwork(inputShape, layers, labels, meta);

        Optimizer? optimizer = null;
        if (expectOptimizer && r.ReadBoolean())
        {
            var options = TrainingOptions.FromDictionary(meta.Training);
            options.Optimizer = (OptimizerKind)r.ReadByte();
            optimizer = Optimizer.Create(options);
            optimizer.ReadState(r);
        }
        return new ModelFileContents(network, optimizer);
    }
}
