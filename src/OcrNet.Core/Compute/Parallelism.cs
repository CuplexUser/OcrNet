namespace OcrNet.Core.Compute;

/// <summary>Process-wide thread budget used by the math kernels and the data pipeline.</summary>
public static class Parallelism
{
    private static int _maxThreads = Environment.ProcessorCount;

    /// <summary>Maximum number of threads the engine may use. Defaults to the logical core count.</summary>
    public static int MaxThreads
    {
        get => _maxThreads;
        set => _maxThreads = Math.Clamp(value, 1, Environment.ProcessorCount * 2);
    }

    public static ParallelOptions Options => new() { MaxDegreeOfParallelism = _maxThreads };
}
