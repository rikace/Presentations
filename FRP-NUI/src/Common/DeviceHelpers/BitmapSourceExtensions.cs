using System.IO;
using System.Security;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DeviceHelpers
{
    public enum ImageFormat
    {
        Png,
        Jpeg,
        Bmp
    }

    public static class BitmapSourceExtensions
    {
        [SecurityCritical]
        public static void Save(this BitmapSource image, string filePath, ImageFormat format)
        {
            BitmapEncoder encoder = null;

            switch (format)
            {
                case ImageFormat.Png:
                    encoder = new PngBitmapEncoder();
                    break;
                case ImageFormat.Jpeg:
                    encoder = new JpegBitmapEncoder();
                    break;
                case ImageFormat.Bmp:
                    encoder = new BmpBitmapEncoder();
                    break;
            }

            if (encoder == null)
                return;

            encoder.Frames.Add(BitmapFrame.Create(BitmapFrame.Create(image)));

            using (var stream = new FileStream(filePath, FileMode.Create))
                encoder.Save(stream);
        }

        public static BitmapSource ToBitmapSource(this byte[] pixels, int width, int height)
        {
            return ToBitmapSource(pixels, width, height, PixelFormats.Bgr32);
        }

        private static BitmapSource ToBitmapSource(this byte[] pixels, int width, int height, PixelFormat format)
        {
            return BitmapSource.Create(width, height, 96, 96, format, null, pixels, width * format.BitsPerPixel / 8);
        }
    }
}