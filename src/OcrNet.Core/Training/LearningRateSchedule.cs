namespace OcrNet.Core.Training;

public static class LearningRateSchedule
{
    /// <summary>Learning rate after <paramref name="epochProgress"/> epochs (fractional) of training.</summary>
    public static float At(TrainingOptions o, double epochProgress)
    {
        float baseLr = o.LearningRate;
        if (o.WarmupEpochs > 0 && epochProgress < o.WarmupEpochs)
            return baseLr * (float)Math.Max(0.01, epochProgress / o.WarmupEpochs);

        double t = epochProgress - Math.Max(0, o.WarmupEpochs);
        switch (o.Schedule)
        {
            case LrScheduleKind.Step:
                return baseLr * MathF.Pow(o.Gamma, (int)(epochProgress / Math.Max(1, o.StepEpochs)));
            case LrScheduleKind.Exponential:
                return baseLr * (float)Math.Pow(o.Gamma, epochProgress);
            case LrScheduleKind.Cosine:
            {
                double span = Math.Max(1e-9, o.Epochs - Math.Max(0, o.WarmupEpochs));
                double progress = Math.Clamp(t / span, 0, 1);
                double min = baseLr * o.MinLearningRateFactor;
                return (float)(min + 0.5 * (baseLr - min) * (1 + Math.Cos(Math.PI * progress)));
            }
            default:
                return baseLr;
        }
    }
}
