using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using System.Drawing;

namespace CsWfAnimations
{
    static class Program
    {
        #region Utilities for running animations

        static bool waiting = true;
        static FrpAnimation af;

        // Show animation and loop until the click
        static void ShowAndWait(Behavior<IDrawing> anim)
        {
            af.Animation = anim;
            waiting = true;
            while (waiting)
            {
                if (!af.Visible) throw new Exception();
                Application.DoEvents();
            }
        }

        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            af = new FrpAnimation();
            af.Click += (sender, e) => { waiting = false; };
            af.Show();

            try { RunAnimation(); } catch { }
            while (af.Visible) Application.DoEvents();
        }

        #endregion

        private static void RunAnimation()
        {
            SimpleDemo(af);
            SolarSystemDemo(af);
        }

        static Behavior<IDrawing> Rotate(this Behavior<IDrawing> img, float dist, float speed)
        {
            var pos = Time.CircularAnim * dist.Forever();
            return img.Translate(pos, pos.Wait(0.5f)).Faster(speed);
        }


        private static void SimpleDemo(FrpAnimation af)
        {
            var greenCircle = Drawings.Circle(Brushes.OliveDrab, 100.0f);
            var drawing =
              greenCircle
               .Translate(-35f, 35f)
               .Compose(greenCircle.Translate(35f, -35f));

            ShowAndWait(Time.Forever(drawing));
        }

        static void SolarSystemDemo(FrpAnimation af)
        {
            var sun = Anims.Cirle(Time.Forever(Brushes.Goldenrod), 100.0f.Forever());
            var earth = Anims.Cirle(Time.Forever(Brushes.SteelBlue), 50.0f.Forever());
            var moon = Anims.Cirle(Time.Forever(Brushes.DimGray), 20.0f.Forever());

            var planets =
               sun.Compose(
                  earth.Compose(moon.Rotate(50.0f, 12.0f))
                       .Rotate(150.0f, 1.0f))
                  .Faster(0.2f);

            ShowAndWait(planets);
        }
    }
}
