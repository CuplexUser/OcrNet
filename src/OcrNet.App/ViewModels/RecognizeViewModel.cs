using System.Collections.ObjectModel;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OcrNet.App.Services;
using OcrNet.Data.Datasets;
using OcrNet.Data.Glyphs;
using OcrNet.Data.Imaging;

namespace OcrNet.App.ViewModels;

public sealed record Prediction(string Label, double Probability, bool IsTop);

public sealed record SegmentResult(BitmapSource Image, string Label, double Confidence);

public enum FontTestOutcome
{
    Correct,
    Wrong,

    /// <summary>The character is not one of the network's classes, so no answer can be right.</summary>
    NotAClass,
}

/// <param name="Glyph">The rendered glyph, kept so a mistake can be added to the training samples.</param>
public sealed record FontTestResult(BitmapSource Image, string Expected, string Predicted, double Confidence, FontTestOutcome Outcome, string Details,
    GrayImage Glyph);

/// <summary>Classifies drawings from the ink canvas (one character, or a line split into characters).</summary>
public sealed partial class RecognizeViewModel : ObservableObject
{
    private readonly AppState _state;
    private GrayImage? _lastDrawing;
    private List<GrayImage> _lastSegments = [];

    public RecognizeViewModel(AppState state)
    {
        _state = state;
        state.NetworkReplaced += (_, _) => Rerun();
        state.WeightsChanged += (_, _) => Rerun();
        _ = LoadFontsAsync();
    }

    public ObservableCollection<Prediction> TopPredictions { get; } = [];

    public ObservableCollection<SegmentResult> Segments { get; } = [];

    public ObservableCollection<string> FontFamilies { get; } = [];

    /// <summary>The font test results currently shown (all, or only the mistakes).</summary>
    public ObservableCollection<FontTestResult> FontTestResults { get; } = [];

    private readonly List<FontTestResult> _fontTestResults = [];

    [ObservableProperty]
    public partial string FontTestSummary { get; set; } = "";

    [ObservableProperty]
    public partial string FontTrainStatus { get; set; } = "";

    /// <summary>Whether the mistakes of the current font test have already been added to the training samples.</summary>
    private bool _mistakesAdded;

    [ObservableProperty]
    public partial bool ShowOnlyMistakes { get; set; }

    [ObservableProperty]
    public partial double PenWidth { get; set; } = 18;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsSingleMode))]
    public partial bool LineMode { get; set; }

    public bool IsSingleMode => !LineMode;

    [ObservableProperty]
    public partial BitmapSource? NormalizedImage { get; set; }

    [ObservableProperty]
    public partial string Recognized { get; set; } = "";

    [ObservableProperty]
    public partial string TeachLabel { get; set; } = "";

    [ObservableProperty]
    public partial string TeachStatus { get; set; } = "";

    [ObservableProperty]
    public partial string? SelectedFontFamily { get; set; }

    [ObservableProperty]
    public partial string FontTestText { get; set; } = "0123456789 OcrNet Räksmörgås";

    public string Hint => _state.HasNetwork
        ? $"Draw a character. The network knows: {string.Join(" ", _state.CharacterSet.Classes.Select(CharacterSet.Display))}"
        : "Build or open a network first.";

    partial void OnLineModeChanged(bool value)
    {
        TeachLabel = "";
        TeachStatus = "";
        Rerun();
    }

    /// <summary>Called by the view with the rasterized canvas whenever the strokes change.</summary>
    public void Recognize(GrayImage? drawing)
    {
        _lastDrawing = drawing;
        _lastSegments = [];
        TopPredictions.Clear();
        Segments.Clear();
        Recognized = "";
        NormalizedImage = null;
        OnPropertyChanged(nameof(Hint));
        if (drawing is null || _state.Network is not { } network)
            return;

        var normalizer = _state.Normalizer;
        int size = normalizer.OutputSize;
        if (size * size != network.InputShape.Size)
            return;

        if (!LineMode)
        {
            var input = Normalizer.Normalize(drawing, normalizer);
            if (drawing.InkBounds() is null)
                return;
            NormalizedImage = BitmapFactory.Grayscale(input, size, size);
            var probabilities = network.Predict(input);
            var ranked = probabilities.Select((p, i) => (p, i)).OrderByDescending(x => x.p).Take(5).ToList();
            foreach (var (p, i) in ranked)
                TopPredictions.Add(new Prediction(CharacterSet.Display(network.Labels[i]), p, i == ranked[0].i));
            Recognized = CharacterSet.Display(network.Labels[ranked[0].i]);
            _state.LastInput = input;
        }
        else
        {
            var text = new System.Text.StringBuilder();
            _lastSegments = Segmenter.SplitCharacters(drawing, minGap: 4);
            foreach (var segment in _lastSegments)
            {
                var input = Normalizer.Normalize(segment, normalizer);
                var probabilities = network.Predict(input);
                int best = Array.IndexOf(probabilities, probabilities.Max());
                string label = network.Labels[best][..1];
                text.Append(label);
                Segments.Add(new SegmentResult(BitmapFactory.Grayscale(input, size, size), label, probabilities[best]));
                _state.LastInput = input;
            }
            Recognized = text.ToString();
        }
    }

    private void Rerun() => Recognize(_lastDrawing);

    /// <summary>
    /// Stores the current drawing as a labelled example of the user's handwriting: the whole drawing
    /// in single-character mode, or each segmented character of a line with its character of the label.
    /// </summary>
    [RelayCommand]
    private void Teach()
    {
        if (LineMode)
            TeachLine();
        else
            TeachCharacter();
    }

    private void TeachCharacter()
    {
        if (_lastDrawing?.InkBounds() is null)
        {
            TeachStatus = "Draw a character first.";
            return;
        }
        string label = TeachLabel.Trim();
        if (label.Length != 1)
        {
            TeachStatus = "Type exactly one character as the correct label.";
            return;
        }
        if (_state.CharacterSet.ClassOf(label[0]) < 0)
        {
            TeachStatus = $"'{label}' is not one of the network's classes.";
            return;
        }
        _state.UserSamples.Add(label[0], _lastDrawing);
        SaveUserSamples();
        int count = _state.UserSamples.Samples.Count(s => s.Character == label[0]);
        TeachStatus = $"Saved as '{label}' ({count} of your drawings for this character, {_state.UserSamples.Count} in total). " +
                      "They are mixed into training the next time you train.";
    }

    private void TeachLine()
    {
        if (_lastSegments.Count == 0)
        {
            TeachStatus = "Write a line of text first.";
            return;
        }
        string label = string.Concat(TeachLabel.Where(c => !char.IsWhiteSpace(c)));
        if (label.Length != _lastSegments.Count)
        {
            TeachStatus = $"The line was split into {_lastSegments.Count} characters but the text has {label.Length}. " +
                          "Type one character per box shown above (spaces are ignored).";
            return;
        }
        var unknown = label.Where(c => _state.CharacterSet.ClassOf(c) < 0).Distinct().ToList();
        if (unknown.Count > 0)
        {
            TeachStatus = $"Not classes of this network: {string.Join(" ", unknown)}";
            return;
        }
        for (int i = 0; i < label.Length; i++)
            _state.UserSamples.Add(label[i], _lastSegments[i]);
        SaveUserSamples();
        TeachStatus = $"Saved {label.Length} characters as '{label}' ({_state.UserSamples.Count} of your drawings in total). " +
                      "They are mixed into training the next time you train.";
    }

    private void SaveUserSamples()
    {
        _state.UserSamples.Save(UserSampleStore.DefaultPath);
        UserSamplesChanged?.Invoke(this, EventArgs.Empty);
    }

    public event EventHandler? UserSamplesChanged;

    private async Task LoadFontsAsync()
    {
        var faces = await _state.GetFontsAsync();
        foreach (var family in faces.Select(f => f.Family).Distinct())
            FontFamilies.Add(family);
        SelectedFontFamily = FontFamilies.FirstOrDefault(f => f == "Segoe Print") ?? FontFamilies.FirstOrDefault();
    }

    /// <summary>Renders each character of the test text in the chosen font and classifies it.</summary>
    [RelayCommand]
    private void TestFont()
    {
        _fontTestResults.Clear();
        FontTestResults.Clear();
        FontTestSummary = "";
        FontTrainStatus = "";
        _mistakesAdded = false;
        if (_state.Network is not { } network || SelectedFontFamily is null)
            return;
        var face = new FontFace(SelectedFontFamily, FontVariant.Regular, false, true, true, false);
        var normalizer = _state.Normalizer;
        int size = normalizer.OutputSize;
        // Render each distinct character once; repeated characters ("sss") reuse the same glyph.
        var glyphs = GlyphRenderer.RenderFace(face, string.Concat(FontTestText.Distinct()))
            .ToDictionary(g => g.Character, g => g.Image);
        var missingGlyphs = new List<char>();
        foreach (char c in FontTestText.Where(c => !char.IsWhiteSpace(c)))
        {
            if (!glyphs.TryGetValue(c, out var image))
            {
                if (!missingGlyphs.Contains(c))
                    missingGlyphs.Add(c);
                continue;
            }
            var input = Normalizer.Normalize(image, normalizer);
            var probabilities = network.Predict(input);
            var ranked = Enumerable.Range(0, probabilities.Length).OrderByDescending(i => probabilities[i]).Take(3).ToArray();
            int best = ranked[0];
            var outcome = !network.Labels.Any(l => l.Contains(c)) ? FontTestOutcome.NotAClass
                : network.Labels[best].Contains(c) ? FontTestOutcome.Correct
                : FontTestOutcome.Wrong;
            string details = outcome == FontTestOutcome.NotAClass
                ? $"'{c}' is not one of this network's classes, so it cannot be recognized."
                : "Top guesses: " + string.Join(" · ", ranked.Select(i => $"{CharacterSet.Display(network.Labels[i])} {probabilities[i]:P0}"));
            _fontTestResults.Add(new FontTestResult(BitmapFactory.Grayscale(input, size, size), c.ToString(),
                CharacterSet.Display(network.Labels[best]), probabilities[best], outcome, details, image));
        }
        FontTestSummary = SummarizeFontTest(missingGlyphs);
        ShowFontTestResults();
        AddMistakesToTrainingCommand.NotifyCanExecuteChanged();
    }

    private bool CanAddMistakesToTraining() => !_mistakesAdded && _fontTestResults.Any(r => r.Outcome == FontTestOutcome.Wrong);

    /// <summary>Saves each misclassified glyph of the font test (once per character) as a training sample.</summary>
    [RelayCommand(CanExecute = nameof(CanAddMistakesToTraining))]
    private void AddMistakesToTraining()
    {
        var mistakes = _fontTestResults.Where(r => r.Outcome == FontTestOutcome.Wrong).DistinctBy(r => r.Expected).ToList();
        foreach (var mistake in mistakes)
            _state.UserSamples.Add(mistake.Expected[0], mistake.Glyph);
        SaveUserSamples();
        _mistakesAdded = true;
        AddMistakesToTrainingCommand.NotifyCanExecuteChanged();
        FontTrainStatus = $"Added {mistakes.Count} glyphs ({string.Join(" ", mistakes.Select(m => m.Expected))}) to your training samples " +
                          $"({_state.UserSamples.Count} in total). Train again on the Training tab to learn them.";
    }

    partial void OnShowOnlyMistakesChanged(bool value) => ShowFontTestResults();

    private void ShowFontTestResults()
    {
        FontTestResults.Clear();
        foreach (var result in _fontTestResults.Where(r => !ShowOnlyMistakes || r.Outcome == FontTestOutcome.Wrong))
            FontTestResults.Add(result);
    }

    private string SummarizeFontTest(List<char> missingGlyphs)
    {
        var known = _fontTestResults.Where(r => r.Outcome != FontTestOutcome.NotAClass).ToList();
        var parts = new List<string>();
        if (known.Count > 0)
        {
            int correct = known.Count(r => r.Outcome == FontTestOutcome.Correct);
            parts.Add($"{correct} of {known.Count} correct ({(double)correct / known.Count:P0})");
        }
        var wrong = _fontTestResults.Where(r => r.Outcome == FontTestOutcome.Wrong)
            .Select(r => $"{r.Expected}→{r.Predicted}").Distinct().ToList();
        if (wrong.Count > 0)
            parts.Add("mistakes: " + string.Join(", ", wrong));
        var notClasses = _fontTestResults.Where(r => r.Outcome == FontTestOutcome.NotAClass).Select(r => r.Expected).Distinct().ToList();
        if (notClasses.Count > 0)
            parts.Add("not classes of this network: " + string.Join(" ", notClasses));
        if (missingGlyphs.Count > 0)
            parts.Add("not in this font: " + string.Join(" ", missingGlyphs));
        return string.Join("  ·  ", parts);
    }
}
