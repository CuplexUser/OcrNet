using System.ComponentModel;
using System.Windows;
using OcrNet.App.Services;
using OcrNet.App.ViewModels;
using OcrNet.App.Views;

namespace OcrNet.App;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        FitToWorkArea();
        DataContext = new MainViewModel();
        Loaded += OnLoaded;
    }

    /// <summary>
    /// Shrinks the default and minimum size to the screen's work area (excluding the taskbar), so the
    /// title bar is always reachable on small or highly scaled displays.
    /// </summary>
    private void FitToWorkArea()
    {
        var area = SystemParameters.WorkArea;
        const double margin = 24;
        Width = Math.Min(Width, area.Width - margin);
        Height = Math.Min(Height, area.Height - margin);
        MinWidth = Math.Min(MinWidth, Width);
        MinHeight = Math.Min(MinHeight, Height);
        if (area.Width < 1280 || area.Height < 760)
            WindowState = WindowState.Maximized;
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        var args = Environment.GetCommandLineArgs();
        int snapshot = Array.IndexOf(args, "--snapshot");
        if (snapshot < 0 || snapshot + 1 >= args.Length)
            return;
        string? Option(string name) => Array.IndexOf(args, name) is var i and >= 0 && i + 1 < args.Length ? args[i + 1] : null;
        await SnapshotRunner.RunAsync(this, (MainViewModel)DataContext, args[snapshot + 1],
            train: args.Contains("--train"),
            recognize: RecognizePage,
            drawText: Option("--draw"),
            fontTestText: Option("--font-test"));
    }

    private void OnClosing(object? sender, CancelEventArgs e)
    {
        var vm = (MainViewModel)DataContext;
        if (vm.State.IsTraining)
        {
            if (MessageDialog.Show("Training is running. Stop it and quit?", "OcrNet",
                    MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes)
            {
                e.Cancel = true;
                return;
            }
            vm.Training.StopCommand.Execute(null);
        }
        if (!vm.ConfirmDiscard())
            e.Cancel = true;
    }
}
