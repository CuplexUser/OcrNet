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
        int draw = Array.IndexOf(args, "--draw");
        await SnapshotRunner.RunAsync(this, (MainViewModel)DataContext, args[snapshot + 1],
            train: args.Contains("--train"),
            draw: text => FindChild<RecognizeView>(this)?.DrawText(text),
            drawText: draw >= 0 && draw + 1 < args.Length ? args[draw + 1] : null);
    }

    private static T? FindChild<T>(DependencyObject parent) where T : DependencyObject
    {
        for (int i = 0; i < System.Windows.Media.VisualTreeHelper.GetChildrenCount(parent); i++)
        {
            var child = System.Windows.Media.VisualTreeHelper.GetChild(parent, i);
            if (child is T match)
                return match;
            if (FindChild<T>(child) is { } nested)
                return nested;
        }
        return null;
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
