namespace OcrNet.Data.Datasets;

/// <summary>The letter coverage of a built-in character set; Swedish letters and symbols are optional extras.</summary>
public enum CharacterSetKind
{
    Digits,
    DigitsAndUpper,
    Balanced,
    Alphanumeric,
}

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

    /// <summary>The Swedish letters beyond A–Z, in alphabetical order.</summary>
    public const string SwedishUpper = "ÅÄÖ";

    public const string SwedishLower = "åäö";

    /// <summary>
    /// Symbols with a distinctive shape. Characters told apart only by size or position (. , ' " _)
    /// are left out: normalization scales and centers every glyph, so they would become identical.
    /// </summary>
    public const string Symbols = "@£$€%&/()?;:!+-=#*";

    public CharacterSet(string name, IEnumerable<string> classes)
        : this(name, classes, null, false, false)
    {
    }

    private CharacterSet(string name, IEnumerable<string> classes, CharacterSetKind? kind, bool hasSwedish, bool hasSymbols)
    {
        Name = name;
        Classes = [.. classes];
        Kind = kind;
        HasSwedish = hasSwedish;
        HasSymbols = hasSymbols;
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

    /// <summary>The built-in coverage, or null for a custom set.</summary>
    public CharacterSetKind? Kind { get; }

    public bool HasSwedish { get; }

    public bool HasSymbols { get; }

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

    public static string KindName(CharacterSetKind kind) => kind switch
    {
        CharacterSetKind.Digits => "Digits",
        CharacterSetKind.DigitsAndUpper => "Digits + uppercase",
        CharacterSetKind.Balanced => "Balanced (merged look-alikes)",
        _ => "Digits + upper + lower",
    };

    /// <summary>
    /// Every built-in combination. Class order is stable: digits, capitals (with merged lowercase),
    /// Swedish capitals, the remaining lowercase letters, Swedish lowercase, symbols, so adding
    /// extras never reorders the classes of an existing set.
    /// </summary>
    public static IReadOnlyList<CharacterSet> Presets { get; } =
    [
        .. from kind in Enum.GetValues<CharacterSetKind>()
           from swedish in kind == CharacterSetKind.Digits ? new[] { false } : new[] { false, true }
           from symbols in new[] { false, true }
           select Build(kind, swedish, symbols),
    ];

    public static CharacterSet DigitsOnly { get; } = Create(CharacterSetKind.Digits);

    public static CharacterSet DigitsAndUpper { get; } = Create(CharacterSetKind.DigitsAndUpper);

    public static CharacterSet Balanced { get; } = Create(CharacterSetKind.Balanced);

    public static CharacterSet Alphanumeric { get; } = Create(CharacterSetKind.Alphanumeric);

    /// <summary>Returns the built-in set for this combination. Digits have no letters, so <paramref name="swedish"/> is ignored for them.</summary>
    public static CharacterSet Create(CharacterSetKind kind, bool swedish = false, bool symbols = false) =>
        Presets.First(p => p.Kind == kind && p.HasSwedish == (swedish && kind != CharacterSetKind.Digits) && p.HasSymbols == symbols);

    private static CharacterSet Build(CharacterSetKind kind, bool swedish, bool symbols)
    {
        static string S(char c) => c.ToString();
        static bool Merged(char upper) => MergedLower.Contains(char.ToLowerInvariant(upper));

        var classes = new List<string>(Digits.Select(S));
        switch (kind)
        {
            case CharacterSetKind.DigitsAndUpper:
                classes.AddRange(Upper.Select(S));
                if (swedish)
                    classes.AddRange(SwedishUpper.Select(S));
                break;
            case CharacterSetKind.Balanced:
                classes.AddRange(Upper.Select(c => Merged(c) ? $"{c}{char.ToLowerInvariant(c)}" : S(c)));
                if (swedish) // Å/å, Ä/ä and Ö/ö differ only in size, which normalization removes.
                    classes.AddRange(SwedishUpper.Select(c => $"{c}{char.ToLowerInvariant(c)}"));
                classes.AddRange(Lower.Where(c => !MergedLower.Contains(c)).Select(S));
                break;
            case CharacterSetKind.Alphanumeric:
                classes.AddRange(Upper.Select(S));
                if (swedish)
                    classes.AddRange(SwedishUpper.Select(S));
                classes.AddRange(Lower.Select(S));
                if (swedish)
                    classes.AddRange(SwedishLower.Select(S));
                break;
        }
        if (symbols)
            classes.AddRange(Symbols.Select(S));

        string name = KindName(kind) + (swedish ? " + Swedish letters" : "") + (symbols ? " + symbols" : "");
        return new CharacterSet(name, classes, kind, swedish, symbols);
    }

    /// <summary>Builds a set with one class per distinct character of <paramref name="characters"/>.</summary>
    public static CharacterSet Custom(string characters) =>
        new("Custom", characters.Where(c => !char.IsWhiteSpace(c)).Distinct().Select(c => c.ToString()));

    /// <summary>Finds the built-in set whose labels equal <paramref name="labels"/>, or builds a custom set from them.</summary>
    public static CharacterSet FromLabels(IReadOnlyList<string> labels) =>
        Presets.FirstOrDefault(p => p.Classes.SequenceEqual(labels)) ?? new CharacterSet("Custom", labels);
}
