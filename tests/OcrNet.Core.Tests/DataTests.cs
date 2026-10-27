using OcrNet.Core.Compute;
using OcrNet.Data.Datasets;
using OcrNet.Data.Imaging;

namespace OcrNet.Core.Tests;

public class DataTests
{
    /// <summary>A filled rectangle with a smaller bar on top, so the shape has a distinct center of mass.</summary>
    private static GrayImage DrawShape(int width, int height, int x0, int y0, int scale)
    {
        var image = new GrayImage(width, height);
        for (int y = 0; y < 6 * scale; y++)
            for (int x = 0; x < 4 * scale; x++)
                image[x0 + x, y0 + y] = 1f;
        for (int y = 0; y < scale; y++)
            for (int x = 0; x < 8 * scale; x++)
                image[x0 + x, y0 + y] = 1f;
        return image;
    }

    [Fact]
    public void Normalizer_IsInvariantToTranslation()
    {
        var options = new NormalizerOptions();
        var a = Normalizer.Normalize(DrawShape(100, 100, 5, 10, 4), options);
        var b = Normalizer.Normalize(DrawShape(100, 100, 50, 40, 4), options);
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i], 1e-5f);
    }

    [Fact]
    public void Normalizer_IsNearlyInvariantToScale()
    {
        var options = new NormalizerOptions();
        var small = Normalizer.Normalize(DrawShape(200, 200, 20, 20, 3), options);
        var large = Normalizer.Normalize(DrawShape(200, 200, 30, 30, 12), options);
        double meanDiff = small.Zip(large, (x, y) => Math.Abs(x - y)).Average();
        Assert.True(meanDiff < 0.03, $"mean absolute difference {meanDiff:0.###}");
    }

    [Fact]
    public void Normalizer_FitsInkIntoBoxAndCentersMass()
    {
        var options = new NormalizerOptions();
        var output = Normalizer.Normalize(DrawShape(120, 90, 30, 20, 5), options);
        var image = new GrayImage(options.OutputSize, options.OutputSize, output);
        var bounds = image.InkBounds()!.Value;
        Assert.True(Math.Max(bounds.Width, bounds.Height) <= options.FitSize + 1);

        double mass = 0, cx = 0, cy = 0;
        for (int y = 0; y < image.Height; y++)
            for (int x = 0; x < image.Width; x++)
            {
                mass += image[x, y];
                cx += x * image[x, y];
                cy += y * image[x, y];
            }
        double center = (options.OutputSize - 1) / 2.0;
        Assert.InRange(cx / mass, center - 1, center + 1);
        Assert.InRange(cy / mass, center - 1, center + 1);
    }

    [Fact]
    public void Normalizer_EmptyImage_GivesZeros()
    {
        var output = Normalizer.Normalize(new GrayImage(40, 40), new NormalizerOptions());
        Assert.All(output, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Segmenter_SplitsCharactersSeparatedByGaps()
    {
        var line = new GrayImage(100, 30);
        foreach (int x0 in new[] { 5, 40, 75 })
            for (int y = 5; y < 25; y++)
                for (int x = x0; x < x0 + 10; x++)
                    line[x, y] = 1f;
        var parts = Segmenter.SplitCharacters(line);
        Assert.Equal(3, parts.Count);
        Assert.All(parts, p => Assert.Equal(10, p.InkBounds()!.Value.Width));
    }

    [Fact]
    public void Augmenter_IsDeterministicForSeed()
    {
        var source = DrawShape(64, 64, 16, 12, 4);
        var options = new AugmentationOptions();
        var a = Augmenter.Distort(source, options, new FastRandom(7));
        var b = Augmenter.Distort(source, options, new FastRandom(7));
        var c = Augmenter.Distort(source, options, new FastRandom(8));
        Assert.Equal(a.Pixels, b.Pixels);
        Assert.NotEqual(a.Pixels, c.Pixels);
    }

    [Fact]
    public void CharacterSets_HaveExpectedSizes()
    {
        Assert.Equal(10, CharacterSet.DigitsOnly.Count);
        Assert.Equal(36, CharacterSet.DigitsAndUpper.Count);
        Assert.Equal(47, CharacterSet.Balanced.Count);
        Assert.Equal(62, CharacterSet.Alphanumeric.Count);
    }

    [Fact]
    public void BalancedSet_MergesLookAlikeCase()
    {
        var set = CharacterSet.Balanced;
        Assert.Equal(set.ClassOf('C'), set.ClassOf('c'));
        Assert.Equal(set.ClassOf('O'), set.ClassOf('o'));
        Assert.NotEqual(set.ClassOf('A'), set.ClassOf('a'));
        Assert.Equal(-1, set.ClassOf('#'));
        Assert.Equal("C/c", CharacterSet.Display(set.Classes[set.ClassOf('c')]));
    }

    [Fact]
    public void FromLabels_RecognizesPresetsAndCustomSets()
    {
        Assert.Same(CharacterSet.Balanced, CharacterSet.FromLabels([.. CharacterSet.Balanced.Classes]));
        var custom = CharacterSet.FromLabels(["+", "-", "="]);
        Assert.Equal("Custom", custom.Name);
        Assert.Equal(3, CharacterSet.Custom("+ - = +").Count);
    }
}
