#:property TargetFramework=net10.0-windows
#:property UseWPF=true
#:property OutputType=Exe
#:property PublishAot=false

// Renders the OcrNet logo (two input neurons feeding an output neuron) into a multi-size .ico.
// Usage: dotnet run tools/make-icon.cs -- src/OcrNet.App/Assets/OcrNet.ico

using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

string output = args.Length > 0 ? args[0] : "OcrNet.ico";
int[] sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256];

var thread = new Thread(() =>
{
    var images = sizes.Select(size => (size, png: RenderPng(size))).ToList();
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    using var file = File.Create(output);
    using var w = new BinaryWriter(file);
    w.Write((ushort)0); // reserved
    w.Write((ushort)1); // type: icon
    w.Write((ushort)images.Count);
    int offset = 6 + 16 * images.Count;
    foreach (var (size, png) in images)
    {
        w.Write((byte)(size >= 256 ? 0 : size));
        w.Write((byte)(size >= 256 ? 0 : size));
        w.Write((byte)0); // palette
        w.Write((byte)0); // reserved
        w.Write((ushort)1); // planes
        w.Write((ushort)32); // bits per pixel
        w.Write(png.Length);
        w.Write(offset);
        offset += png.Length;
    }
    foreach (var (_, png) in images)
        w.Write(png);
});
thread.SetApartmentState(ApartmentState.STA);
thread.Start();
thread.Join();
Console.WriteLine($"Wrote {output} ({string.Join(", ", sizes)} px)");

static byte[] RenderPng(int size)
{
    // Designed on a 32-unit grid; strokes get relatively thicker at small sizes so they stay visible.
    double s = size / 32.0;
    double stroke = size <= 20 ? 2.6 : size <= 32 ? 2.3 : 2.0;
    var background = new LinearGradientBrush(Color.FromRgb(0x26, 0x2D, 0x37), Color.FromRgb(0x14, 0x18, 0x1D), 90);
    var border = new SolidColorBrush(Color.FromRgb(0x34, 0x3C, 0x48));
    var muted = new SolidColorBrush(Color.FromRgb(0xA7, 0xB0, 0xBD));
    var signal = new SolidColorBrush(Color.FromRgb(0x2E, 0xC4, 0xB6));
    var negative = new SolidColorBrush(Color.FromRgb(0xFF, 0x6B, 0x57));

    var visual = new DrawingVisual();
    using (var dc = visual.RenderOpen())
    {
        dc.PushTransform(new ScaleTransform(s, s));
        dc.DrawRoundedRectangle(background, size >= 32 ? new Pen(border, 1) : null, new Rect(1, 1, 30, 30), 7, 7);
        var a = new Point(9.5, 10);
        var b = new Point(9.5, 22);
        var output = new Point(23, 16);
        dc.DrawLine(new Pen(signal, stroke) { StartLineCap = PenLineCap.Round, EndLineCap = PenLineCap.Round }, a, output);
        dc.DrawLine(new Pen(negative, stroke) { StartLineCap = PenLineCap.Round, EndLineCap = PenLineCap.Round }, b, output);
        dc.DrawEllipse(muted, null, a, 3.4, 3.4);
        dc.DrawEllipse(muted, null, b, 3.4, 3.4);
        dc.DrawEllipse(signal, null, output, 4.2, 4.2);
        dc.Pop();
    }

    var bitmap = new RenderTargetBitmap(size, size, 96, 96, PixelFormats.Pbgra32);
    bitmap.Render(visual);
    var encoder = new PngBitmapEncoder();
    encoder.Frames.Add(BitmapFrame.Create(bitmap));
    using var stream = new MemoryStream();
    encoder.Save(stream);
    return stream.ToArray();
}
