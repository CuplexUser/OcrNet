using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace OcrNet.App.Controls;

/// <summary>Collapsed when the value is true, visible otherwise (the opposite of <see cref="BooleanToVisibilityConverter"/>).</summary>
public sealed class InverseBoolToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is true ? Visibility.Collapsed : Visibility.Visible;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is not Visibility.Visible;
}
