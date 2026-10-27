using OcrNet.Core.Layers;
using OcrNet.Core.Serialization;

namespace OcrNet.Core.Optimizers;

/// <summary>Stable ids (stored in model files with the optimizer state).</summary>
public enum OptimizerKind : byte
{
    Sgd = 1,
    RmsProp = 2,
    Adam = 3,
    AdamW = 4,
}

/// <summary>
/// Base class for gradient-descent optimizers. Per-parameter state (momentum, moment estimates)
/// is kept in arrays that line up with the parameter list passed to <see cref="Step"/>.
/// </summary>
public abstract class Optimizer
{
    private readonly List<float[][]> _state = [];

    protected Optimizer(float weightDecay)
    {
        WeightDecay = weightDecay;
    }

    public abstract OptimizerKind Kind { get; }

    /// <summary>L2 penalty (SGD, RMSProp, Adam) or decoupled decay (AdamW); applied to weights only.</summary>
    public float WeightDecay { get; }

    /// <summary>Number of update steps taken so far.</summary>
    public long StepCount { get; private set; }

    /// <summary>Number of state arrays the optimizer keeps per parameter.</summary>
    protected abstract int StateSlots { get; }

    public void Step(IReadOnlyList<Parameter> parameters, float learningRate)
    {
        EnsureState(parameters);
        StepCount++;
        for (int i = 0; i < parameters.Count; i++)
            Update(parameters[i], _state[i], learningRate);
    }

    protected abstract void Update(Parameter parameter, float[][] state, float learningRate);

    private void EnsureState(IReadOnlyList<Parameter> parameters)
    {
        if (_state.Count == parameters.Count)
            return;
        _state.Clear();
        foreach (var p in parameters)
        {
            var slots = new float[StateSlots][];
            for (int s = 0; s < slots.Length; s++)
                slots[s] = new float[p.Length];
            _state.Add(slots);
        }
    }

    public void WriteState(BinaryWriter writer)
    {
        writer.Write(StepCount);
        writer.Write(_state.Count);
        foreach (var slots in _state)
        {
            writer.Write(slots.Length);
            foreach (var arr in slots)
                writer.WriteFloats(arr);
        }
    }

    public void ReadState(BinaryReader reader)
    {
        StepCount = reader.ReadInt64();
        int count = reader.ReadInt32();
        _state.Clear();
        for (int i = 0; i < count; i++)
        {
            var slots = new float[reader.ReadInt32()][];
            for (int s = 0; s < slots.Length; s++)
                slots[s] = reader.ReadFloats();
            _state.Add(slots);
        }
    }

    public static Optimizer Create(Training.TrainingOptions options) => options.Optimizer switch
    {
        OptimizerKind.Sgd => new SgdOptimizer(options.Momentum, options.Nesterov, options.WeightDecay),
        OptimizerKind.RmsProp => new RmsPropOptimizer(options.RmsPropRho, options.Epsilon, options.WeightDecay),
        OptimizerKind.Adam => new AdamOptimizer(options.Beta1, options.Beta2, options.Epsilon, options.WeightDecay, decoupled: false),
        OptimizerKind.AdamW => new AdamOptimizer(options.Beta1, options.Beta2, options.Epsilon, options.WeightDecay, decoupled: true),
        _ => throw new NotSupportedException(options.Optimizer.ToString()),
    };
}
