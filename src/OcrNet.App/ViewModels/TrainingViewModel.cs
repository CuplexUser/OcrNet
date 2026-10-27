using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using OcrNet.App.Views;
using OcrNet.App.Controls;
using OcrNet.App.Services;
using OcrNet.Core.Network;
using OcrNet.Core.Optimizers;
using OcrNet.Core.Training;
using OcrNet.Data.Datasets;
using OcrNet.Data.Imaging;

namespace OcrNet.App.ViewModels;

public sealed record MisclassifiedItem(BitmapSource Image, string Text);

/// <summary>Hyperparameters, the training loop's lifecycle and its live charts.</summary>
public sealed partial class TrainingViewModel : ObservableObject
{
    private readonly AppState _state;
    private CancellationTokenSource? _cts;
    private Trainer? _trainer;
    private int _firstEpoch = 1;

    public TrainingViewModel(AppState state)
    {
        _state = state;
        state.NetworkReplaced += (_, _) =>
        {
            OnPropertyChanged(nameof(Options));
            RebuildCharts();
            Confusion = null;
            Worst.Clear();
        };
        state.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is nameof(AppState.IsTraining) or nameof(AppState.Network))
            {
                OnPropertyChanged(nameof(IsTraining));
                OnPropertyChanged(nameof(CanStart));
                StartCommand.NotifyCanExecuteChanged();
                ContinueCommand.NotifyCanExecuteChanged();
                StopCommand.NotifyCanExecuteChanged();
                PauseCommand.NotifyCanExecuteChanged();
                EvaluateMnistCommand.NotifyCanExecuteChanged();
            }
        };
        RebuildCharts();
    }

    public TrainingOptions Options => _state.TrainingOptions;

    /// <summary>Call when <see cref="AppState.TrainingOptions"/> was replaced (e.g. by a preset).</summary>
    public void OptionsReplaced()
    {
        OnPropertyChanged(nameof(Options));
        OnPropertyChanged(nameof(DropoutOverrideEnabled));
        OnPropertyChanged(nameof(DropoutOverride));
    }

    public IReadOnlyList<OptimizerKind> Optimizers { get; } = Enum.GetValues<OptimizerKind>();

    public IReadOnlyList<LrScheduleKind> Schedules { get; } = Enum.GetValues<LrScheduleKind>();

    public int MaxThreads { get; } = Environment.ProcessorCount;

    public bool IsTraining => _state.IsTraining;

    public bool CanStart => !_state.IsTraining && _state.HasNetwork;

    public bool DropoutOverrideEnabled
    {
        get => Options.DropoutOverride >= 0;
        set
        {
            Options.DropoutOverride = value ? 0.2f : -1f;
            OnPropertyChanged();
            OnPropertyChanged(nameof(DropoutOverride));
        }
    }

    public float DropoutOverride
    {
        get => Math.Max(0, Options.DropoutOverride);
        set
        {
            if (DropoutOverrideEnabled)
                Options.DropoutOverride = Math.Clamp(value, 0f, 0.9f);
            OnPropertyChanged();
        }
    }

    [ObservableProperty]
    public partial bool SaveOptimizerState { get; set; } = true;

    [ObservableProperty]
    public partial bool IsPaused { get; set; }

    [ObservableProperty]
    public partial string Phase { get; set; } = "Idle";

    [ObservableProperty]
    public partial double EpochProgress { get; set; }

    [ObservableProperty]
    public partial double OverallProgress { get; set; }

    [ObservableProperty]
    public partial string EpochText { get; set; } = "–";

    [ObservableProperty]
    public partial string LossText { get; set; } = "–";

    [ObservableProperty]
    public partial string AccuracyText { get; set; } = "–";

    [ObservableProperty]
    public partial string ValidationText { get; set; } = "–";

    [ObservableProperty]
    public partial string LearningRateText { get; set; } = "–";

    [ObservableProperty]
    public partial string SpeedText { get; set; } = "–";

    [ObservableProperty]
    public partial string TimeText { get; set; } = "–";

    [ObservableProperty]
    public partial string RemainingText { get; set; } = "";

    [ObservableProperty]
    public partial IReadOnlyList<ChartSeries> LossSeries { get; set; } = [];

    [ObservableProperty]
    public partial IReadOnlyList<ChartSeries> AccuracySeries { get; set; } = [];

    [ObservableProperty]
    public partial IReadOnlyList<ChartSeries> LearningRateSeries { get; set; } = [];

    [ObservableProperty]
    public partial int[,]? Confusion { get; set; }

    public IReadOnlyList<string>? ConfusionLabels => _state.Network?.Labels;

    public ObservableCollection<MisclassifiedItem> Worst { get; } = [];

    public ObservableCollection<string> Log { get; } = [];

    private bool CanContinue() => CanStart && _state.Network!.Metadata.History.Count > 0;

    /// <summary>Re-initializes the weights (same architecture) and trains from scratch.</summary>
    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartAsync()
    {
        var current = _state.Network!;
        if (current.Metadata.History.Count > 0 &&
            MessageDialog.Show("Training from scratch discards the current weights and history. Continue?",
                "OcrNet", MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes)
            return;

        var metadata = new ModelMetadata { Name = current.Metadata.Name, Description = current.Metadata.Description };
        var fresh = NetworkBuilder.Build(new NetworkSpec
        {
            InputShape = current.InputShape,
            Labels = [.. current.Labels],
            HiddenLayers = NetworkBuilder.DescribeHiddenLayers(current),
            Seed = Options.Seed,
        }, metadata);
        _state.SetNetwork(fresh, optimizer: null, _state.ModelPath);
        _state.IsDirty = true;
        await RunAsync(resume: false);
    }

    /// <summary>Trains the current weights for more epochs, keeping the optimizer state.</summary>
    [RelayCommand(CanExecute = nameof(CanContinue))]
    private Task ContinueAsync() => RunAsync(resume: true);

    [RelayCommand(CanExecute = nameof(IsTraining))]
    private void Pause()
    {
        if (_trainer is null)
            return;
        if (_trainer.IsPaused)
        {
            _trainer.Resume();
            IsPaused = false;
            Phase = "Training";
        }
        else
        {
            _trainer.Pause();
            IsPaused = true;
            Phase = "Paused";
        }
    }

    [RelayCommand(CanExecute = nameof(IsTraining))]
    private void Stop()
    {
        _cts?.Cancel();
        _trainer?.Resume();
        Phase = "Stopping…";
    }

    private async Task RunAsync(bool resume)
    {
        var network = _state.Network!;
        _cts = new CancellationTokenSource();
        _state.IsTraining = true;
        IsPaused = false;
        Phase = "Preparing data";
        try
        {
            var status = new Progress<string>(s => Phase = s);
            var data = await _state.BuildDatasetAsync(status, _cts.Token);
            network.Metadata.Preprocessing = _state.DatasetOptions.Normalizer.ToDictionary();
            AddLog($"Dataset: {data.TrainFontCount} training faces, {data.ValidationFontCount} held-out faces " +
                   $"({data.ValidationSampleCount:N0} validation samples), {data.UserSampleCount} user drawings, " +
                   $"{data.TrainSamplesPerEpoch:N0} augmented samples per epoch.");

            Phase = "Preparing validation set";
            await Task.Run(() => data.GetValidationBatches(Math.Max(Options.BatchSize, 256)), _cts.Token);

            var options = Options.Clone();
            AddLog($"{(resume ? "Continuing" : "Starting")}: {options.Epochs} epochs, {options.Optimizer}, lr {options.LearningRate:G3}, " +
                   $"batch {options.BatchSize}, weight decay {options.WeightDecay:G3}, {options.Threads} threads.");
            Phase = "Training";
            _firstEpoch = network.Metadata.History.Count + 1;
            _trainer = new Trainer(network, data, options, new Progress<TrainingProgress>(OnProgress), resume ? _state.Optimizer : null);
            var result = await _trainer.RunAsync(_cts.Token);

            _state.Optimizer = result.Optimizer;
            _state.IsDirty = true;
            if (result.FinalValidation is { } validation)
                ShowValidation(validation);
            string outcome = result.Diverged ? "stopped: the loss diverged (try a lower learning rate)"
                : result.Cancelled ? "stopped by user"
                : result.StoppedEarly ? "stopped early (no improvement)"
                : "finished";
            AddLog($"Training {outcome} after {result.Duration:mm\\:ss}. Best epoch {result.BestEpoch}" +
                   (result.FinalValidation is { } v ? $", validation accuracy {v.Accuracy:P2}." : "."));
            Phase = $"Done – {outcome}";
            _state.Status = $"Training {outcome}";
        }
        catch (OperationCanceledException)
        {
            Phase = "Cancelled";
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException)
        {
            Phase = "Failed";
            AddLog("Error: " + ex.Message);
            MessageDialog.Show(ex.Message, "OcrNet – training", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally
        {
            _trainer = null;
            _state.IsTraining = false;
            IsPaused = false;
            _state.NotifyWeightsChanged();
        }
    }

    private void OnProgress(TrainingProgress p)
    {
        EpochProgress = (double)p.BatchIndex / Math.Max(1, p.BatchesPerEpoch);
        OverallProgress = (p.Epoch - _firstEpoch + EpochProgress) / Math.Max(1, p.Epochs - _firstEpoch + 1);
        EpochText = $"{p.Epoch} / {p.Epochs}";
        LossText = p.Loss.ToString("0.0000");
        AccuracyText = p.Accuracy.ToString("P2");
        LearningRateText = p.LearningRate.ToString("G3");
        SpeedText = $"{p.SamplesPerSecond:N0}/s";
        TimeText = $"{p.Elapsed:mm\\:ss}";
        RemainingText = p.Remaining is { } r ? $"≈{r:mm\\:ss} left" : "";

        if (p.CompletedEpoch is { } e)
        {
            _state.History.Add(e);
            ValidationText = float.IsNaN(e.ValidationAccuracy) ? "–" : e.ValidationAccuracy.ToString("P2");
            AddLog($"Epoch {e.Epoch}: loss {e.TrainLoss:0.000}, train {e.TrainAccuracy:P1}, " +
                   $"validation {e.ValidationAccuracy:P2} (loss {e.ValidationLoss:0.000}), lr {e.LearningRate:G3}, {e.Seconds:0.0}s");
            RebuildCharts();
            if (p.Validation is { } validation)
                ShowValidation(validation);
            _state.NotifyWeightsChanged();
        }
    }

    private void ShowValidation(EvaluationResult validation)
    {
        _state.LastValidation = validation;
        OnPropertyChanged(nameof(ConfusionLabels));
        Confusion = validation.Confusion;
        ValidationText = validation.Accuracy.ToString("P2");
        var labels = _state.Network!.Labels;
        int size = _state.Network.InputShape.W;
        Worst.Clear();
        foreach (var m in validation.Worst.Take(32))
            Worst.Add(new MisclassifiedItem(BitmapFactory.Grayscale(m.Input, size, size),
                $"{CharacterSet.Display(labels[m.Expected])}→{CharacterSet.Display(labels[m.Predicted])} {m.Confidence:P0}"));
    }

    private void RebuildCharts()
    {
        var signal = (Brush)Application.Current.FindResource("SignalBrush");
        var validation = (Brush)Application.Current.FindResource("ValidationBrush");
        var highlight = (Brush)Application.Current.FindResource("HighlightBrush");
        var trainLoss = new ChartSeries { Name = "train", Stroke = signal };
        var valLoss = new ChartSeries { Name = "validation", Stroke = validation };
        var trainAcc = new ChartSeries { Name = "train", Stroke = signal };
        var valAcc = new ChartSeries { Name = "validation", Stroke = validation };
        var lr = new ChartSeries { Name = "learning rate", Stroke = highlight };
        foreach (var h in _state.History)
        {
            trainLoss.Points.Add(new Point(h.Epoch, h.TrainLoss));
            valLoss.Points.Add(new Point(h.Epoch, h.ValidationLoss));
            trainAcc.Points.Add(new Point(h.Epoch, h.TrainAccuracy * 100));
            valAcc.Points.Add(new Point(h.Epoch, h.ValidationAccuracy * 100));
            lr.Points.Add(new Point(h.Epoch, h.LearningRate));
        }
        LossSeries = [trainLoss, valLoss];
        AccuracySeries = [trainAcc, valAcc];
        LearningRateSeries = [lr];
    }

    private void AddLog(string message)
    {
        Log.Insert(0, $"{DateTime.Now:HH:mm:ss}  {message}");
        while (Log.Count > 300)
            Log.RemoveAt(Log.Count - 1);
    }

    /// <summary>Accuracy of the current model on MNIST-format IDX files (real handwritten digits).</summary>
    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task EvaluateMnistAsync()
    {
        var network = _state.Network!;
        var set = _state.CharacterSet;
        if (set.ClassOf('0') < 0)
        {
            MessageDialog.Show("MNIST contains digits; the current network has no digit classes.", "OcrNet");
            return;
        }
        var images = new OpenFileDialog { Title = "MNIST images (e.g. t10k-images-idx3-ubyte[.gz])", Filter = "IDX images|*idx3*|All files|*.*" };
        if (images.ShowDialog(Application.Current.MainWindow) != true)
            return;
        var labels = new OpenFileDialog { Title = "MNIST labels (e.g. t10k-labels-idx1-ubyte[.gz])", Filter = "IDX labels|*idx1*|All files|*.*", InitialDirectory = System.IO.Path.GetDirectoryName(images.FileName) };
        if (labels.ShowDialog(Application.Current.MainWindow) != true)
            return;

        Phase = "Evaluating on MNIST…";
        var normalizer = _state.Normalizer;
        try
        {
            var (accuracy, count) = await Task.Run(() =>
            {
                var (imgs, lbls) = MnistReader.Read(images.FileName, labels.FileName);
                int correct = 0;
                var input = new float[network.InputShape.Size];
                for (int i = 0; i < imgs.Count; i++)
                {
                    Normalizer.Normalize(imgs[i], normalizer, input);
                    var probabilities = network.Predict(input);
                    int predicted = Array.IndexOf(probabilities, probabilities.Max());
                    if (network.Labels[predicted].Contains((char)('0' + lbls[i])))
                        correct++;
                }
                return ((double)correct / imgs.Count, imgs.Count);
            });
            AddLog($"MNIST evaluation: {accuracy:P2} correct on {count:N0} handwritten digits.");
            Phase = $"MNIST accuracy {accuracy:P2}";
        }
        catch (Exception ex) when (ex is System.IO.IOException or System.IO.InvalidDataException)
        {
            Phase = "MNIST evaluation failed";
            MessageDialog.Show(ex.Message, "OcrNet – MNIST", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }
}
