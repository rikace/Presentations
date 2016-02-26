using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CsWfAnimations
{
    public partial class FrpAnimation : Form
    {

        private Behavior<IDrawing> _animation;
        public Behavior<IDrawing> Animation
        {
            set { startTime = DateTime.UtcNow; _animation = value; Invalidate(); }
            get { return _animation; }
        }

        private Timer timer;
        private DateTime startTime;

        public FrpAnimation()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint |
                ControlStyles.Opaque | ControlStyles.OptimizedDoubleBuffer, true);

            startTime = DateTime.UtcNow;
            this.timer = new Timer();
            timer.Interval = 25;
            timer.Tick += (s, e) => this.Invalidate();
            InitializeComponent();
            timer.Start();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            e.Graphics.FillRectangle(Brushes.White, new Rectangle(new Point(0, 0), ClientSize));
            e.Graphics.TranslateTransform(ClientSize.Width / 2, ClientSize.Height / 2);
            if (Animation != null)
            {
                var time = (DateTime.UtcNow - startTime).TotalSeconds;
                var drawing = Animation.BehaviorFunc(new BehaviorContext((float)time));
                drawing.Draw(e.Graphics);
            }
            base.OnPaint(e);
        }
    }
}