using System.Runtime.InteropServices;

namespace OcrNet.Core.Serialization;

/// <summary>Bulk float I/O and small helpers shared by the model file and optimizer state.</summary>
public static class BinaryExtensions
{
    public static void WriteFloats(this BinaryWriter writer, ReadOnlySpan<float> values)
    {
        writer.Write(values.Length);
        if (BitConverter.IsLittleEndian)
        {
            writer.Write(MemoryMarshal.AsBytes(values));
        }
        else
        {
            foreach (float v in values)
                writer.Write(v);
        }
    }

    public static float[] ReadFloats(this BinaryReader reader, int maxLength = 256 * 1024 * 1024)
    {
        int length = reader.ReadInt32();
        if (length < 0 || length > maxLength)
            throw new InvalidDataException($"Invalid array length {length}.");
        var values = new float[length];
        if (BitConverter.IsLittleEndian)
        {
            reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
        }
        else
        {
            for (int i = 0; i < length; i++)
                values[i] = reader.ReadSingle();
        }
        return values;
    }

    public static void WriteDictionary(this BinaryWriter writer, IReadOnlyDictionary<string, string> values)
    {
        writer.Write(values.Count);
        foreach (var (key, value) in values)
        {
            writer.Write(key);
            writer.Write(value);
        }
    }

    public static Dictionary<string, string> ReadStringDictionary(this BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var result = new Dictionary<string, string>(count);
        for (int i = 0; i < count; i++)
            result[reader.ReadString()] = reader.ReadString();
        return result;
    }
}
