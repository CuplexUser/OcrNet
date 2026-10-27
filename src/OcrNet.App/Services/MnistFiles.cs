using System.IO;
using System.Windows;
using Microsoft.Win32;
using OcrNet.App.Views;
using OcrNet.Data.Datasets;

namespace OcrNet.App.Services;

/// <summary>Lets the user choose a pair of MNIST IDX files (images, plus labels when they cannot be found by name).</summary>
public static class MnistFiles
{
    /// <returns>The chosen images and labels paths, or null if the user cancelled.</returns>
    public static (string Images, string Labels)? Pick(string title, string? initialDirectory)
    {
        var images = new OpenFileDialog
        {
            Title = title,
            Filter = "IDX images (*idx3*)|*idx3*|All files|*.*",
            InitialDirectory = initialDirectory is not null && Directory.Exists(initialDirectory) ? initialDirectory : "",
        };
        if (images.ShowDialog(Application.Current.MainWindow) != true)
            return null;
        // The labels normally sit next to the images; only ask when they cannot be found by name.
        if (MnistReader.FindLabelsFile(images.FileName) is { } found)
            return (images.FileName, found);

        MessageDialog.Show($"No labels file matching {Path.GetFileName(images.FileName)} (e.g. t10k-labels-idx1-ubyte) was found next to it. " +
                           "Choose it in the next window.", "OcrNet – MNIST", MessageBoxButton.OK, MessageBoxImage.Information);
        var labels = new OpenFileDialog
        {
            Title = "MNIST labels (e.g. t10k-labels-idx1-ubyte[.gz])",
            Filter = "IDX labels (*idx1*)|*idx1*|All files|*.*",
            InitialDirectory = Path.GetDirectoryName(images.FileName),
        };
        return labels.ShowDialog(Application.Current.MainWindow) == true ? (images.FileName, labels.FileName) : null;
    }
}
