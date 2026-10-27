using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using OcrNet.App.ViewModels;

namespace OcrNet.App.Services;

/// <summary>
/// Developer smoke test: <c>OcrNet.App.exe --snapshot &lt;folder&gt; [--train] [--draw &lt;text&gt;]</c>
/// optionally trains the default network briefly, draws a sample, renders every tab offscreen to
/// PNG files and exits. Works without an interactive desktop (e.g. a locked session or CI).
/// </summary>
public static class SnapshotRunner
{
    public static async Task RunAsync(Window window, MainViewModel vm, string folder, bool train, Action<string>? draw, string? drawText)
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
        if (draw is not null && drawText is not null)
        {
            vm.SelectedTab = 3;
            await Idle();
            draw(drawText);
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
        vm.State.IsDirty = false; // nothing to save prompts about in a smoke test
        window.Close();
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
