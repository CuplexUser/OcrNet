using OcrNet.Core.Tensors;

namespace OcrNet.Core.Layers;

/// <summary>Recreates layers from their kind and serialized configuration.</summary>
public static class LayerFactory
{
    public static Layer Create(LayerKind kind, TensorShape inputShape, BinaryReader config) => kind switch
    {
        LayerKind.Dense => DenseLayer.ReadConfig(inputShape, config),
        LayerKind.Activation => ActivationLayer.ReadConfig(inputShape, config),
        LayerKind.Dropout => DropoutLayer.ReadConfig(inputShape, config),
        LayerKind.BatchNorm1D => BatchNorm1DLayer.ReadConfig(inputShape, config),
        _ => throw new NotSupportedException(
            $"Layer type {kind} ({(ushort)kind}) is not supported by this version of OcrNet."),
    };
}
