namespace OcrNet.Data.Glyphs;

[Flags]
public enum FontVariant
{
    Regular = 0,
    Bold = 1,
    Italic = 2,
    BoldItalic = Bold | Italic,
}

/// <summary>An installed font family in one style, identified by plain strings so it can cross threads.</summary>
public sealed record FontFace(string Family, FontVariant Variant, bool IsSimulated, bool HasLowercase, bool HasDescenders, bool IsDecorative)
{
    /// <summary>Stable key used by the glyph cache and the dataset settings.</summary>
    public string Key => $"{Family}|{Variant}";

    public string DisplayName => Variant == FontVariant.Regular ? Family : $"{Family} {Variant}";

    /// <summary>
    /// All-caps / small-caps fonts draw lowercase letters as capitals; their lowercase glyphs
    /// would teach the network the wrong shapes, so they are only used for digits and capitals.
    /// </summary>
    public bool UseLowercase => HasLowercase && HasDescenders;

    /// <summary>Ornate, outline, stencil or blackletter fonts; not selected for training by default.</summary>
    /// <remarks>Set from <see cref="FontCatalog.DecorativeFamilies"/>.</remarks>
    public bool SelectedByDefault => !IsDecorative;
}
