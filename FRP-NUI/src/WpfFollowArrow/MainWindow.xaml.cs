using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WpfFollowArrow
{

    public partial class MainWindow : Window
    {
        double widthChar = 20;
        double windowWidth = 800;
        double windowHeight = 400;

        FollowArrowRX.FollowArrowListener listener;
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            this.Width = windowWidth;
            this.Height = windowHeight;

            var chars = (from c in "Reactive Extensions are awsome!"
                         select new TextBlock
                         {
                             Width = widthChar,
                             Height = 30.0,
                             FontSize = 20.0,
                             Text = c.ToString(),
                             Foreground = Brushes.Black,
                             Background = Brushes.Transparent
                         }).ToArray();

            this.Title = "";
            foreach (var tb in chars)
                canvas.Children.Add(tb);

            listener = new FollowArrowRX.FollowArrowListener(this, chars);
            listener.Start(this);
        }
    }
}
