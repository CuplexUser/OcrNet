using System.Windows;
using System.Windows.Media;

namespace OcrNet.Data.Glyphs;

/// <summary>Discovers installed fonts that can be used as training material.</summary>
public static class FontCatalog
{
    /// <summary>Name fragments of icon, symbol and barcode fonts whose "letters" are not letters.</summary>
    private static readonly string[] Blocklist =
    [
        "Wingdings", "Webdings", "Marlett", "Symbol", "MDL2", "Fluent Icons", "Emoji", "Bookshelf",
        "MT Extra", "Barcode", "Dingbat", "Ornament", "Icons", "Glyph", "Holo", "Font Awesome",
        "Material", "Zapf", "Code39", "Code 128", "OCR-A", "Braille", "Marlett",
    ];

    /// <summary>
    /// Fonts whose letterforms are far from handwriting (ornate scripts, outline/inline, stencil,
    /// blackletter, extreme display faces). They are listed but unselected by default.
    /// </summary>
    public static readonly HashSet<string> DecorativeFamilies = new(StringComparer.OrdinalIgnoreCase)
    {
        "Algerian", "Bauhaus 93", "Blackadder ITC", "Bodoni MT Poster", "Broadway", "Brush Script MT", "Castellar",
        "Chiller", "Colonna MT", "Curlz MT", "Edwardian Script ITC", "Engravers MT", "Felix Titling", "Forte",
        "Freestyle Script", "French Script MT", "Gabriola", "Gigi", "Goudy Stout", "Haettenschweiler", "Harlow Solid",
        "Imprint MT Shadow", "Informal Roman", "Jokerman", "Juice ITC", "Kunstler Script", "Magneto",
        "Matura MT Script Capitals", "Mistral", "Niagara Engraved", "Niagara Solid", "OCR A", "Old English Text MT",
        "Onyx", "Palace Script MT", "Parchment", "Playbill", "Pristina", "Rage", "Ravie", "Script MT",
        "Showcard Gothic", "Snap ITC", "Stencil", "Vivaldi", "Vladimir Script", "Wide Latin",
    };

    private const string RequiredCharacters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private const string Lowercase = "abcdefghijklmnopqrstuvwxyz";

    /// <summary>Enumerates usable font faces for the requested style variants, sorted by name.</summary>
    public static List<FontFace> Discover(IEnumerable<FontVariant> variants)
    {
        var variantList = variants.Distinct().ToList();
        var result = new List<FontFace>();
        var seen = new HashSet<string>();

        foreach (var family in Fonts.SystemFontFamilies)
        {
            string name = family.Source;
            if (name.StartsWith("Global ", StringComparison.Ordinal) || Blocklist.Any(b => name.Contains(b, StringComparison.OrdinalIgnoreCase)))
                continue;

            foreach (var variant in variantList)
            {
                var typeface = CreateTypeface(family, variant);
                if (!typeface.TryGetGlyphTypeface(out var glyphs) || glyphs.Symbol)
                    continue;
                if (!RequiredCharacters.All(c => glyphs.CharacterToGlyphMap.ContainsKey(c)))
                    continue;

                // Several requested variants can resolve to the same physical face; keep the first.
                string identity = $"{glyphs.FontUri}|{glyphs.StyleSimulations}";
                if (!seen.Add(identity))
                    continue;

                bool hasLowercase = Lowercase.All(c => glyphs.CharacterToGlyphMap.ContainsKey(c))
                    && Lowercase.Any(c => glyphs.CharacterToGlyphMap[c] != glyphs.CharacterToGlyphMap[char.ToUpperInvariant(c)]);
                bool hasDescenders = hasLowercase && HasDescenders(glyphs);
                result.Add(new FontFace(name, variant, glyphs.StyleSimulations != StyleSimulations.None, hasLowercase, hasDescenders,
                    DecorativeFamilies.Contains(name)));
            }
        }

        result.Sort((a, b) => string.Compare(a.DisplayName, b.DisplayName, StringComparison.OrdinalIgnoreCase));
        return result;
    }

    internal static Typeface CreateTypeface(FontFamily family, FontVariant variant) => new(
        family,
        variant.HasFlag(FontVariant.Italic) ? FontStyles.Italic : FontStyles.Normal,
        variant.HasFlag(FontVariant.Bold) ? FontWeights.Bold : FontWeights.Normal,
        FontStretches.Normal);

    /// <summary>True when 'g', 'p', 'q' and 'y' reach below the baseline (i.e. not a small-caps font).</summary>
    private static bool HasDescenders(GlyphTypeface glyphs)
    {
        const double em = 100;
        int descending = 0;
        foreach (char c in "gpqy")
        {
            var outline = glyphs.GetGlyphOutline(glyphs.CharacterToGlyphMap[c], em, em);
            if (!outline.Bounds.IsEmpty && outline.Bounds.Bottom > em * 0.08)
                descending++;
        }
        return descending >= 3;
    }
}
