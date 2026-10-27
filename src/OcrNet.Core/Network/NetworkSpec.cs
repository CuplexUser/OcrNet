using OcrNet.Core.Init;
using OcrNet.Core.Layers;
using OcrNet.Core.Tensors;

namespace OcrNet.Core.Network;

/// <summary>Description of one hidden block: Dense → [BatchNorm] → activation → Dropout.</summary>
public sealed class HiddenLayerSpec
{
    public int Units { get; set; } = 256;

    public ActivationKind Activation { get; set; } = ActivationKind.ReLU;

    public float Dropout { get; set; } = 0.2f;

    public bool BatchNorm { get; set; }

    public HiddenLayerSpec Clone() => (HiddenLayerSpec)MemberwiseClone();
}

/// <summary>Everything needed to build a fresh, randomly initialized network.</summary>
public sealed class NetworkSpec
{
    public TensorShape InputShape { get; set; } = new(1, 28, 28);

    public List<HiddenLayerSpec> HiddenLayers { get; set; } = [];

    public List<string> Labels { get; set; } = [];

    public WeightInitKind WeightInit { get; set; } = WeightInitKind.Auto;

    public ulong Seed { get; set; } = 42;
}
