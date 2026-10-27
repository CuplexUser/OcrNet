namespace OcrNet.Data.Datasets;

/// <summary>
/// The classes a network distinguishes. Each class is a string of one or more characters that
/// share an output (e.g. "Oo" when upper- and lowercase o are merged). The class string is also the
/// label stored in the model file.
/// </summary>
public sealed class CharacterSet
{
    public const string Digits = "0123456789";
    public const string Upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    public const string Lower = "abcdefghijklmnopqrstuvwxyz";

    /// <summary>Lowercase letters that look like their capitals when handwritten (EMNIST "balanced" merge).</summary>
    public const string MergedLower = "cijklmopsuvwxyz";

    public CharacterSet(string name, IEnumerable<string> classes)
    {
        Name = name;
        Classes = [.. classes];
        if (Classes.Count < 2)
            throw new ArgumentException("A character set needs at least two classes.", nameof(classes));
        var seen = new HashSet<char>();
        foreach (char c in Classes.SelectMany(s => s))
            if (!seen.Add(c))
                throw new ArgumentException($"Character '{c}' appears in more than one class.", nameof(classes));
    }

    public string Name { get; }

    public IReadOnlyList<string> Classes { get; }

    public int Count => Classes.Count;

    /// <summary>All characters used by the set.</summary>
    public string Characters => string.Concat(Classes);

    public int ClassOf(char c)
    {
        for (int i = 0; i < Classes.Count; i++)
            if (Classes[i].Contains(c))
                return i;
        return -1;
    }

    public override string ToString() => $"{Name} ({Count} classes)";

    /// <summary>Human-readable form of a class label: "C/c" for merged classes.</summary>
    public static string Display(string label) => label.Length <= 1 ? label : string.Join("/", label.ToCharArray());

    public static CharacterSet DigitsOnly { get; } = new("Digits", Digits.Select(c => c.ToString()));

    public static CharacterSet DigitsAndUpper { get; } = new("Digits + uppercase", (Digits + Upper).Select(c => c.ToString()));

    public static CharacterSet Alphanumeric { get; } = new("Digits + upper + lower", (Digits + Upper + Lower).Select(c => c.ToString()));

    public static CharacterSet Balanced { get; } = new("Balanced (47, merged look-alikes)",
        Digits.Select(c => c.ToString())
            .Concat(Upper.Select(c => MergedLower.Contains(char.ToLowerInvariant(c)) ? $"{c}{char.ToLowerInvariant(c)}" : c.ToString()))
            .Concat(Lower.Where(c => !MergedLower.Contains(c)).Select(c => c.ToString())));

    public static IReadOnlyList<CharacterSet> Presets { get; } = [DigitsOnly, DigitsAndUpper, Balanced, Alphanumeric];

    /// <summary>Builds a set with one class per distinct character of <paramref name="characters"/>.</summary>
    public static CharacterSet Custom(string characters) =>
        new("Custom", characters.Where(c => !char.IsWhiteSpace(c)).Distinct().Select(c => c.ToString()));

    /// <summary>Finds the preset whose labels equal <paramref name="labels"/>, or builds a custom set from them.</summary>
    public static CharacterSet FromLabels(IReadOnlyList<string> labels) =>
        Presets.FirstOrDefault(p => p.Classes.SequenceEqual(labels)) ?? new CharacterSet("Custom", labels);
}
