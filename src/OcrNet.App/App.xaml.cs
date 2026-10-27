using System.Globalization;
using System.Windows;
using System.Windows.Markup;
using System.Windows.Threading;

using OcrNet.App.Views;
namespace OcrNet.App;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        // The UI is English; use one number format everywhere (WPF bindings default to en-US anyway).
        var culture = CultureInfo.GetCultureInfo("en-US");
        CultureInfo.DefaultThreadCurrentCulture = culture;
        CultureInfo.DefaultThreadCurrentUICulture = culture;
        CultureInfo.CurrentCulture = culture;
        CultureInfo.CurrentUICulture = culture;
        FrameworkElement.LanguageProperty.OverrideMetadata(typeof(FrameworkElement),
            new FrameworkPropertyMetadata(XmlLanguage.GetLanguage(culture.IetfLanguageTag)));
        base.OnStartup(e);
        DispatcherUnhandledException += OnUnhandledException;
    }

    private static void OnUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        MessageDialog.Show(e.Exception.Message, "OcrNet – unexpected error", MessageBoxButton.OK, MessageBoxImage.Error);
        e.Handled = true;
    }
}
