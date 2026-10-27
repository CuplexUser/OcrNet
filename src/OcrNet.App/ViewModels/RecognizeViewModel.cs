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

/// <summary>Classifies drawings from the ink canvas (one character, or a line split into characters).</summary>
public sealed partial class RecognizeViewModel : ObservableObject
{
    private readonly AppState _state;
    private GrayImage? _lastDrawing;

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

    public ObservableCollection<SegmentResult> FontTestResults { get; } = [];

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
    public partial string FontTestText { get; set; } = "0123456789 OcrNet";

    public string Hint => _state.HasNetwork
        ? $"Draw a character. The network knows: {string.Join(" ", _state.CharacterSet.Classes.Select(CharacterSet.Display))}"
        : "Build or open a network first.";

    partial void OnLineModeChanged(bool value) => Rerun();

    /// <summary>Called by the view with the rasterized canvas whenever the strokes change.</summary>
    public void Recognize(GrayImage? drawing)
    {
        _lastDrawing = drawing;
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
            foreach (var segment in Segmenter.SplitCharacters(drawing, minGap: 4))
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

    /// <summary>Stores the current drawing as a labelled example of the user's handwriting.</summary>
    [RelayCommand]
    private void Teach()
    {
        if (_lastDrawing?.InkBounds() is null || LineMode)
        {
            TeachStatus = "Draw a single character first.";
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
        _state.UserSamples.Save(UserSampleStore.DefaultPath);
        int count = _state.UserSamples.Samples.Count(s => s.Character == label[0]);
        TeachStatus = $"Saved as '{label}' ({count} of your drawings for this character, {_state.UserSamples.Count} in total). " +
                      "They are mixed into training the next time you train.";
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
        FontTestResults.Clear();
        if (_state.Network is not { } network || SelectedFontFamily is null)
            return;
        var face = new FontFace(SelectedFontFamily, FontVariant.Regular, false, true, true, false);
        var normalizer = _state.Normalizer;
        int size = normalizer.OutputSize;
        var glyphs = GlyphRenderer.RenderFace(face, FontTestText).ToDictionary(g => g.Character, g => g.Image);
        foreach (char c in FontTestText.Where(c => !char.IsWhiteSpace(c)))
        {
            if (!glyphs.TryGetValue(c, out var image))
                continue;
            var input = Normalizer.Normalize(image, normalizer);
            var probabilities = network.Predict(input);
            int best = Array.IndexOf(probabilities, probabilities.Max());
            string predicted = CharacterSet.Display(network.Labels[best]);
            bool correct = network.Labels[best].Contains(c);
            FontTestResults.Add(new SegmentResult(BitmapFactory.Grayscale(input, size, size),
                $"{c} → {predicted}{(correct ? "" : "  ✗")}", probabilities[best]));
        }
    }
}
