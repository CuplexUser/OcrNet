using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using OcrNet.App.Services;
using OcrNet.Core.Serialization;

namespace OcrNet.App.ViewModels;

public sealed partial class MainViewModel : ObservableObject
{
    private const string FileFilter = "OcrNet model (*.ocrnet)|*.ocrnet|All files|*.*";

    public MainViewModel()
    {
        State = new AppState();
        Designer = new DesignerViewModel(State);
        Dataset = new DatasetViewModel(State);
        Training = new TrainingViewModel(State);
        Recognize = new RecognizeViewModel(State);
        Visualize = new VisualizeViewModel(State);

        Designer.TrainingOptionsReplaced += (_, _) => Training.OptionsReplaced();
        Recognize.UserSamplesChanged += (_, _) => Dataset.RefreshUserSamples();
        State.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is nameof(AppState.IsTraining) or nameof(AppState.Network))
            {
                SaveCommand.NotifyCanExecuteChanged();
                SaveAsCommand.NotifyCanExecuteChanged();
                OpenCommand.NotifyCanExecuteChanged();
            }
        };

        // Start with an untrained digits network so every tab has something to show.
        Designer.BuildCommand.Execute(null);
        State.IsDirty = false;
        State.Status = "Pick a preset on the Design tab, then train on the Training tab.";
    }

    public AppState State { get; }

    public DesignerViewModel Designer { get; }

    public DatasetViewModel Dataset { get; }

    public TrainingViewModel Training { get; }

    public RecognizeViewModel Recognize { get; }

    public VisualizeViewModel Visualize { get; }

    [ObservableProperty]
    public partial int SelectedTab { get; set; }

    partial void OnSelectedTabChanged(int value)
    {
        if (value == 4)
            Visualize.Refresh(weightsChanged: false);
    }

    private bool CanUseFiles() => !State.IsTraining;

    private bool CanSave() => CanUseFiles() && State.HasNetwork;

    [RelayCommand(CanExecute = nameof(CanUseFiles))]
    private void Open()
    {
        if (!ConfirmDiscard())
            return;
        var dialog = new OpenFileDialog { Filter = FileFilter, Title = "Open model" };
        if (dialog.ShowDialog() != true)
            return;
        try
        {
            var contents = ModelFile.Load(dialog.FileName);
            State.SetNetwork(contents.Network, contents.Optimizer, dialog.FileName);
            State.IsDirty = false;
            var accuracy = contents.Network.Metadata.Metrics.TryGetValue("ValidationAccuracy", out double a) ? $", validation accuracy {a:P2}" : "";
            State.Status = $"Opened {Path.GetFileName(dialog.FileName)}: {contents.Network}{accuracy}";
            SelectedTab = 3;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException or ArgumentException)
        {
            MessageBox.Show($"Could not open the model:\n{ex.Message}", "OcrNet", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    [RelayCommand(CanExecute = nameof(CanSave))]
    private void Save()
    {
        if (State.ModelPath is null)
            SaveAs();
        else
            SaveTo(State.ModelPath);
    }

    [RelayCommand(CanExecute = nameof(CanSave))]
    private void SaveAs()
    {
        var dialog = new SaveFileDialog
        {
            Filter = FileFilter,
            Title = "Save model",
            FileName = SafeFileName(State.Network!.Metadata.Name) + ModelFile.Extension,
        };
        if (dialog.ShowDialog() == true)
            SaveTo(dialog.FileName);
    }

    private void SaveTo(string path)
    {
        try
        {
            ModelFile.Save(path, State.Network!, Training.SaveOptimizerState ? State.Optimizer : null);
            State.ModelPath = path;
            State.IsDirty = false;
            State.Status = $"Saved {Path.GetFileName(path)} ({new FileInfo(path).Length / 1024.0:N0} KB)";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            MessageBox.Show($"Could not save the model:\n{ex.Message}", "OcrNet", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    /// <summary>Asks before discarding unsaved training; returns false to cancel.</summary>
    public bool ConfirmDiscard()
    {
        if (!State.IsDirty || State.Network?.Metadata.History.Count is null or 0)
            return true;
        var answer = MessageBox.Show("Save the trained network before continuing?", "OcrNet",
            MessageBoxButton.YesNoCancel, MessageBoxImage.Question);
        if (answer == MessageBoxResult.Yes)
            Save();
        return answer != MessageBoxResult.Cancel && !(answer == MessageBoxResult.Yes && State.IsDirty);
    }

    private static string SafeFileName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        string cleaned = new([.. name.Select(c => invalid.Contains(c) ? '_' : c)]);
        return string.IsNullOrWhiteSpace(cleaned) ? "model" : cleaned.Trim();
    }
}
