using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OcrNet.App.Views;
using OcrNet.App.Services;
using OcrNet.Data.Datasets;
using OcrNet.Data.Glyphs;
using OcrNet.Data.Imaging;

namespace OcrNet.App.ViewModels;

/// <summary>One installed font family in the font list.</summary>
public sealed partial class FontItem(string family, bool isDecorative, bool hasLowercase, string variants) : ObservableObject
{
    public string Family { get; } = family;

    public bool IsDecorative { get; } = isDecorative;

    public bool HasLowercase { get; } = hasLowercase;

    public string Variants { get; } = variants;

    public string Notes => string.Join(" · ", new[]
    {
        IsDecorative ? "decorative" : null,
        HasLowercase ? null : "no lowercase",
    }.Where(n => n is not null));

    [ObservableProperty]
    public partial bool IsSelected { get; set; }
}

public sealed record PreviewSample(BitmapSource Image, string Label);

/// <summary>Font selection, augmentation settings and a live preview of generated training samples.</summary>
public sealed partial class DatasetViewModel : ObservableObject
{
    private readonly AppState _state;
    private bool _bulkUpdate;

    public DatasetViewModel(AppState state)
    {
        _state = state;
        FontsView = CollectionViewSource.GetDefaultView(Fonts);
        FontsView.Filter = o => o is FontItem f &&
            (string.IsNullOrWhiteSpace(FontFilter) || f.Family.Contains(FontFilter, StringComparison.OrdinalIgnoreCase));
        state.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is nameof(AppState.MnistTraining) or nameof(AppState.Network))
            {
                OnPropertyChanged(nameof(HasMnist));
                OnPropertyChanged(nameof(MnistSummary));
                RemoveMnistCommand.NotifyCanExecuteChanged();
            }
        };
        _ = LoadFontsAsync();
        _ = state.RestoreMnistTrainingAsync();
    }

    public DatasetOptions Options => _state.DatasetOptions;

    public AugmentationOptions Augmentation => _state.DatasetOptions.Augmentation;

    public ObservableCollection<FontItem> Fonts { get; } = [];

    public ICollectionView FontsView { get; }

    public ObservableCollection<PreviewSample> PreviewSamples { get; } = [];

    [ObservableProperty]
    public partial string FontFilter { get; set; } = "";

    [ObservableProperty]
    public partial string FontSummary { get; set; } = "Discovering installed fonts…";

    [ObservableProperty]
    public partial string PreviewStatus { get; set; } = "";

    [ObservableProperty]
    public partial bool IsBusy { get; set; }

    public bool UseRegular { get => Has(FontVariant.Regular); set => SetVariant(FontVariant.Regular, value); }

    public bool UseBold { get => Has(FontVariant.Bold); set => SetVariant(FontVariant.Bold, value); }

    public bool UseItalic { get => Has(FontVariant.Italic); set => SetVariant(FontVariant.Italic, value); }

    public bool UseBoldItalic { get => Has(FontVariant.BoldItalic); set => SetVariant(FontVariant.BoldItalic, value); }

    public int UserSampleCount => _state.UserSamples.Count;

    public string UserSampleSummary
    {
        get
        {
            var counts = _state.UserSamples.CountByCharacter();
            return counts.Count == 0
                ? "No drawings collected yet. Use “Teach” on the Recognize tab to add your own handwriting."
                : $"{_state.UserSamples.Count} drawings: " +
                  string.Join("  ", counts.OrderBy(c => c.Key).Select(c => $"{c.Key}×{c.Value}"));
        }
    }

    public bool HasMnist => _state.MnistTraining is not null;

    public string MnistSummary => _state.MnistTraining switch
    {
        null => "Not loaded. Add the MNIST training set (train-images-idx3-ubyte, 60,000 handwritten digits) to train on real handwriting, " +
                "then verify with the t10k test set.",
        var m when _state.CharacterSet.ClassOf('0') < 0 =>
            $"{m.Count:N0} digits from {System.IO.Path.GetFileName(m.ImagesPath)}, but the current network has no digit classes, so they are not used.",
        var m => $"{m.Count:N0} handwritten digits from {System.IO.Path.GetFileName(m.ImagesPath)}. " +
                 (MnistReader.FindTestImagesFile(m.ImagesPath) is not null
                     ? "The matching t10k test set was found; it is checked automatically after each training run."
                     : "No matching t10k test set was found next to it; verify with “Evaluate on MNIST…” on the Training tab."),
    };

    private bool Has(FontVariant v) => _state.SelectedVariants.Contains(v);

    private void SetVariant(FontVariant variant, bool on)
    {
        if (on)
            _state.SelectedVariants.Add(variant);
        else
            _state.SelectedVariants.Remove(variant);
        OnPropertyChanged(variant switch
        {
            FontVariant.Bold => nameof(UseBold),
            FontVariant.Italic => nameof(UseItalic),
            FontVariant.BoldItalic => nameof(UseBoldItalic),
            _ => nameof(UseRegular),
        });
        _ = UpdateSummaryAsync();
    }

    partial void OnFontFilterChanged(string value) => FontsView.Refresh();

    private async Task LoadFontsAsync()
    {
        var faces = await _state.GetFontsAsync();
        foreach (var group in faces.GroupBy(f => f.Family))
        {
            var first = group.First();
            var item = new FontItem(group.Key, first.IsDecorative, group.Any(f => f.UseLowercase),
                string.Join(", ", group.Select(f => f.Variant.ToString())))
            {
                IsSelected = first.SelectedByDefault,
            };
            item.PropertyChanged += (_, e) =>
            {
                if (e.PropertyName == nameof(FontItem.IsSelected) && !_bulkUpdate)
                    SyncSelection();
            };
            Fonts.Add(item);
        }
        SyncSelection();
    }

    private void SyncSelection()
    {
        _state.SelectedFontKeys = [.. Fonts.Where(f => f.IsSelected).Select(f => f.Family)];
        _ = UpdateSummaryAsync();
    }

    private async Task UpdateSummaryAsync()
    {
        var selected = await _state.GetSelectedFontsAsync();
        int families = selected.Select(f => f.Family).Distinct().Count();
        FontSummary = $"{families} of {Fonts.Count} families selected · {selected.Count} font faces with the chosen styles";
    }

    [RelayCommand]
    private void SelectAll() => SetAll(_ => true);

    [RelayCommand]
    private void SelectNone() => SetAll(_ => false);

    [RelayCommand]
    private void SelectDefaults() => SetAll(f => !f.IsDecorative);

    private void SetAll(Func<FontItem, bool> selector)
    {
        _bulkUpdate = true;
        foreach (var f in Fonts)
            f.IsSelected = selector(f);
        _bulkUpdate = false;
        SyncSelection();
    }

    [RelayCommand]
    private async Task RefreshPreviewAsync()
    {
        if (IsBusy)
            return;
        IsBusy = true;
        try
        {
            var data = await _state.BuildDatasetAsync(new Progress<string>(s => PreviewStatus = s), CancellationToken.None);
            var samples = await Task.Run(() => data.Preview(60, (ulong)Environment.TickCount64));
            var set = _state.CharacterSet;
            int size = data.InputShape.W;
            PreviewSamples.Clear();
            foreach (var (input, label) in samples)
                PreviewSamples.Add(new PreviewSample(BitmapFactory.Grayscale(input, size, size), CharacterSet.Display(set.Classes[label])));
            PreviewStatus = $"{data.TrainFontCount} training faces ({data.TrainGlyphCount:N0} glyphs) · " +
                            $"{data.ValidationFontCount} held-out faces for validation ({data.ValidationSampleCount:N0} samples) · " +
                            $"{data.UserSampleCount} of your drawings" +
                            (data.MnistSampleCount > 0 ? $" · {data.MnistSampleCount:N0} MNIST digits" : "");
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException)
        {
            PreviewStatus = ex.Message;
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private void ClearUserSamples()
    {
        if (MessageDialog.Show("Delete all of your collected drawings?", "OcrNet", MessageBoxButton.YesNo, MessageBoxImage.Warning) != MessageBoxResult.Yes)
            return;
        _state.UserSamples.Clear();
        _state.UserSamples.Save(UserSampleStore.DefaultPath);
        RefreshUserSamples();
    }

    [RelayCommand]
    private async Task LoadMnistAsync()
    {
        if (MnistFiles.Pick("MNIST training images (e.g. train-images-idx3-ubyte[.gz])", null) is not var (images, labels))
            return;
        if (System.IO.Path.GetFileName(images).Contains("t10k", StringComparison.OrdinalIgnoreCase) &&
            MessageDialog.Show("This looks like the MNIST test set. Training on it leaves nothing unseen to verify with; " +
                               "the training set is train-images-idx3-ubyte. Use it anyway?",
                "OcrNet – MNIST", MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes)
            return;
        try
        {
            IsBusy = true;
            var set = await Task.Run(() => MnistReader.Read(images, labels));
            _state.SetMnistTraining(set, labels);
        }
        catch (Exception ex) when (ex is System.IO.IOException or System.IO.InvalidDataException or UnauthorizedAccessException)
        {
            MessageDialog.Show(ex.Message, "OcrNet – MNIST", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand(CanExecute = nameof(HasMnist))]
    private void RemoveMnist() => _state.SetMnistTraining(null, null);

    public void RefreshUserSamples()
    {
        OnPropertyChanged(nameof(UserSampleCount));
        OnPropertyChanged(nameof(UserSampleSummary));
    }
}
