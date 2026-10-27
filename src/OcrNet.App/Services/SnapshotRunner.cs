using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using OcrNet.App.ViewModels;

namespace OcrNet.App.Services;

/// <summary>
/// Developer smoke test: <c>OcrNet.App.exe --snapshot &lt;folder&gt; [--train] [--draw &lt;text&gt;] [--font-test &lt;text&gt;]</c>
/// optionally trains the default network briefly, draws a sample, runs the font test, renders every
/// tab offscreen to PNG files, checks that Esc clears the canvas (checks.txt), captures line mode at the minimum width and exits. Works without an interactive desktop (e.g. a locked session or CI).
/// </summary>
public static class SnapshotRunner
{
    public static async Task RunAsync(Window window, MainViewModel vm, string folder, bool train, Views.RecognizeView? recognize,
        string? drawText, string? fontTestText)
    {
        Directory.CreateDirectory(folder);
        await Idle();
        if (train)
        {
            vm.State.TrainingOptions.Epochs = 3;
            vm.Training.OptionsReplaced();
            await vm.Training.StartCommand.ExecuteAsync(null);
            File.WriteAllLines(Path.Combine(folder, "training-log.txt"), vm.Training.Log.Reverse());
        }
        if (fontTestText is not null)
        {
            vm.Recognize.FontTestText = fontTestText;
            vm.Recognize.TestFontCommand.Execute(null);
        }
        if (recognize is not null && drawText is not null)
        {
            vm.SelectedTab = 3;
            await Idle();
            recognize.DrawText(drawText);
            await Idle();
        }

        for (int tab = 0; tab < 5; tab++)
        {
            vm.SelectedTab = tab;
            await Idle();
            await Task.Delay(400);
            await Idle();
            Save(window, Path.Combine(folder, $"tab{tab + 1}.png"));
            if (FindScrollContent(window) is { ActualHeight: > 0 } full && full.ActualHeight > ((FrameworkElement)window.Content).ActualHeight)
                SaveElement(full, window.Background, Path.Combine(folder, $"tab{tab + 1}-full.png"));
        }

        if (recognize is not null && drawText is not null)
        {
            await CheckEscapeClearsCanvasAsync(window, vm, recognize, drawText, Path.Combine(folder, "checks.txt"));

            // Line mode at the narrowest window size, where the layout is tightest.
            window.WindowState = WindowState.Normal;
            window.Width = window.MinWidth;
            vm.Recognize.LineMode = true;
            await Idle();
            recognize.DrawText(drawText);
            await Idle();
            await Task.Delay(400);
            await Idle();
            Save(window, Path.Combine(folder, "tab4-line-narrow.png"));
        }

        var dialog = new Views.MessageDialog("The current network has unsaved training. Replace it with a new, untrained network?",
            "OcrNet", MessageBoxButton.YesNoCancel, MessageBoxImage.Question) { Owner = window, ShowActivated = false };
        dialog.Show();
        await Idle();
        Save(dialog, Path.Combine(folder, "dialog.png"));
        dialog.Close();

        vm.State.IsDirty = false; // nothing to save prompts about in a smoke test
        window.Close();
    }

    /// <summary>
    /// Sends Esc from outside the Recognize view (the tab strip) and from the canvas itself, the way
    /// it arrives after drawing with the mouse, and records whether the canvas was cleared each time.
    /// </summary>
    private static async Task CheckEscapeClearsCanvasAsync(Window window, MainViewModel vm, Views.RecognizeView recognize, string drawText, string path)
    {
        vm.SelectedTab = 3;
        await Idle();
        var lines = new List<string>();
        var sources = new (string Name, UIElement Element)[]
        {
            ("tab strip", FindDescendant<System.Windows.Controls.TabControl>(window)!),
            ("canvas", recognize.Canvas),
        };
        foreach (var (name, element) in sources)
        {
            recognize.DrawText(drawText);
            await Idle();
            int before = recognize.StrokeCount;
            PressKey(window, element, System.Windows.Input.Key.Escape);
            await Idle();
            int after = recognize.StrokeCount;
            lines.Add($"Esc from {name} clears canvas: {(before > 0 && after == 0 ? "PASS" : "FAIL")} (strokes {before} -> {after})");
        }
        File.WriteAllLines(path, lines);
    }

    /// <summary>Raises PreviewKeyDown then (unless handled) KeyDown on <paramref name="target"/>, like a real key press.</summary>
    private static void PressKey(Window window, UIElement target, System.Windows.Input.Key key)
    {
        var source = PresentationSource.FromVisual(window)!;
        var preview = new System.Windows.Input.KeyEventArgs(System.Windows.Input.Keyboard.PrimaryDevice, source, 0, key)
        {
            RoutedEvent = System.Windows.Input.Keyboard.PreviewKeyDownEvent,
        };
        target.RaiseEvent(preview);
        if (preview.Handled)
            return;
        target.RaiseEvent(new System.Windows.Input.KeyEventArgs(System.Windows.Input.Keyboard.PrimaryDevice, source, 0, key)
        {
            RoutedEvent = System.Windows.Input.Keyboard.KeyDownEvent,
        });
    }

    private static T? FindDescendant<T>(DependencyObject root) where T : DependencyObject
    {
        for (int i = 0; i < VisualTreeHelper.GetChildrenCount(root); i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            if (child is T match)
                return match;
            if (FindDescendant<T>(child) is { } nested)
                return nested;
        }
        return null;
    }

    private static async Task Idle() =>
        await Application.Current.Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);

    /// <summary>The content of the selected tab's outermost scrolled viewer, if any (to capture what is scrolled out of view).</summary>
    private static FrameworkElement? FindScrollContent(DependencyObject root)
    {
        var queue = new Queue<DependencyObject>([root]);
        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            if (node is System.Windows.Controls.ScrollViewer { Content: FrameworkElement content } viewer && viewer.ExtentHeight > viewer.ViewportHeight + 1)
                return content;
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(node); i++)
                queue.Enqueue(VisualTreeHelper.GetChild(node, i));
        }
        return null;
    }

    private static void SaveElement(FrameworkElement element, Brush background, string path)
    {
        var bitmap = new RenderTargetBitmap((int)element.ActualWidth, (int)element.ActualHeight, 96, 96, PixelFormats.Pbgra32);
        var visual = new DrawingVisual();
        using (var dc = visual.RenderOpen())
        {
            var bounds = new Rect(0, 0, element.ActualWidth, element.ActualHeight);
            dc.DrawRectangle(background, null, bounds);
            dc.DrawRectangle(new VisualBrush(element) { Stretch = Stretch.None, AlignmentX = AlignmentX.Left, AlignmentY = AlignmentY.Top }, null, bounds);
        }
        bitmap.Render(visual);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using var file = File.Create(path);
        encoder.Save(file);
    }

    private static void Save(Window window, string path)
    {
        var content = (FrameworkElement)window.Content;
        double dpi = VisualTreeHelper.GetDpi(window).PixelsPerDip;
        var bitmap = new RenderTargetBitmap(
            (int)(content.ActualWidth * dpi), (int)(content.ActualHeight * dpi), 96 * dpi, 96 * dpi, PixelFormats.Pbgra32);
        var background = new DrawingVisual();
        using (var dc = background.RenderOpen())
            dc.DrawRectangle(window.Background, null, new Rect(0, 0, content.ActualWidth, content.ActualHeight));
        bitmap.Render(background);
        bitmap.Render(content);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using var file = File.Create(path);
        encoder.Save(file);
    }
}
