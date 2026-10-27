namespace OcrNet.Data.Imaging;

/// <summary>Splits a line of handwriting into single characters by vertical projection.</summary>
public static class Segmenter
{
    /// <summary>
    /// Returns the column ranges that contain ink, separated by at least <paramref name="minGap"/>
    /// empty columns. Dots of 'i'/'j' stay with their stem because they share columns.
    /// </summary>
    public static List<GrayImage> SplitCharacters(GrayImage image, int minGap = 3, float threshold = 0.1f)
    {
        var inkColumns = new bool[image.Width];
        for (int x = 0; x < image.Width; x++)
            for (int y = 0; y < image.Height; y++)
                if (image[x, y] > threshold)
                {
                    inkColumns[x] = true;
                    break;
                }

        var segments = new List<GrayImage>();
        int start = -1, gap = 0;
        for (int x = 0; x <= image.Width; x++)
        {
            bool ink = x < image.Width && inkColumns[x];
            if (ink)
            {
                if (start < 0)
                    start = x;
                gap = 0;
            }
            else if (start >= 0)
            {
                gap++;
                if (gap >= minGap || x == image.Width)
                {
                    int end = x - gap + 1;
                    segments.Add(image.Crop(start, 0, end - start, image.Height));
                    start = -1;
                    gap = 0;
                }
            }
        }
        return segments;
    }
}
