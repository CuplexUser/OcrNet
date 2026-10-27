using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace OcrNet.App.Views;

/// <summary>
/// Themed replacement for <see cref="MessageBox"/>: same buttons and results, but it matches the
/// app's dark theme and carries the app icon. Use <see cref="Show"/>.
/// </summary>
public partial class MessageDialog : Window
{
    private MessageBoxResult _result;

    internal MessageDialog(string message, string title, MessageBoxButton buttons, MessageBoxImage image)
    {
        InitializeComponent();
        Title = title;
        MessageText.Text = message;

        (string glyph, string brush)? icon = image switch
        {
            MessageBoxImage.Error => ("\uEA39", "NegativeBrush"),
            MessageBoxImage.Warning => ("\uE7BA", "HighlightBrush"),
            MessageBoxImage.Question => ("\uE9CE", "ValidationBrush"),
            MessageBoxImage.Information => ("\uE946", "SignalBrush"),
            _ => null,
        };
        if (icon is { } i)
        {
            Glyph.Text = i.glyph;
            Glyph.Foreground = (Brush)FindResource(i.brush);
            Glyph.Visibility = Visibility.Visible;
        }

        // The result when the dialog is closed with Esc or the title-bar button.
        _result = buttons switch
        {
            MessageBoxButton.YesNo => MessageBoxResult.No,
            MessageBoxButton.OKCancel or MessageBoxButton.YesNoCancel => MessageBoxResult.Cancel,
            _ => MessageBoxResult.OK,
        };

        var choices = buttons switch
        {
            MessageBoxButton.OKCancel => new[] { MessageBoxResult.OK, MessageBoxResult.Cancel },
            MessageBoxButton.YesNo => [MessageBoxResult.Yes, MessageBoxResult.No],
            MessageBoxButton.YesNoCancel => [MessageBoxResult.Yes, MessageBoxResult.No, MessageBoxResult.Cancel],
            _ => [MessageBoxResult.OK],
        };
        foreach (var choice in choices)
        {
            bool primary = choice == choices[0];
            var button = new Button
            {
                Content = choice.ToString(),
                MinWidth = 88,
                Margin = new Thickness(8, 0, 0, 0),
                IsDefault = primary,
                IsCancel = choice == _result,
            };
            if (primary)
                button.SetResourceReference(StyleProperty, "AccentButtonStyle");
            button.Click += (_, _) =>
            {
                _result = choice;
                Close();
            };
            Buttons.Children.Add(button);
        }
    }

    internal MessageBoxResult Result => _result;

    /// <summary>Shows a modal message over the main window and returns the chosen button.</summary>
    public static MessageBoxResult Show(string message, string title = "OcrNet",
        MessageBoxButton buttons = MessageBoxButton.OK, MessageBoxImage image = MessageBoxImage.None)
    {
        var dialog = new MessageDialog(message, title, buttons, image);
        if (Application.Current?.MainWindow is { IsLoaded: true } owner && owner != dialog)
            dialog.Owner = owner;
        else
            dialog.WindowStartupLocation = WindowStartupLocation.CenterScreen;
        dialog.ShowDialog();
        return dialog.Result;
    }
}
