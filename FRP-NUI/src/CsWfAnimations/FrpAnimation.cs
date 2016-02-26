using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace CsWfAnimations
{
    internal struct BehaviorContext
    {
        public BehaviorContext(float time)
        {
            this.time = time;
        }
        private readonly float time;
        // Gets current time of the animation
        public float Time { get { return time; } }
    }

    public class Behavior<T>
    {
        internal Behavior(Func<BehaviorContext, T> f)
        {
            this.f = f;
        }
        private Func<BehaviorContext, T> f;
        // Function that calculates the value
        internal Func<BehaviorContext, T> BehaviorFunc { get { return f; } }


        Behavior<float> ToSingle()
        {
            return Behavior.Create(ctx => Convert.ToSingle(this.BehaviorFunc(ctx)));
        }
        public static Behavior<float> operator *(Behavior<T> a, Behavior<float> b)
        {
            return Behavior.Lift((float x, float y) => x * y)(a.ToSingle(), b);
        }
        public static Behavior<float> operator +(Behavior<T> a, Behavior<float> b)
        {
            return Behavior.Lift((float x, float y) => x + y)(a.ToSingle(), b);
        }
    }

    // Creating behavior from a function 
    internal static class Behavior
    {
        internal static Behavior<T> Create<T>(Func<BehaviorContext, T> f)
        {
            return new Behavior<T>(f);
        }

        // Implementing lifting and map (select)
        public static Behavior<R> Selelct<T, R>(this Behavior<T> behavior, Func<T, R> f)
        {
            // Return a behavior that applies the function
            return Lift(f)(behavior);

        }

        public static Func<Behavior<T>, Behavior<R>> Lift<T, R>(Func<T, R> f)
        {
            return (b) => Create(ctx => f(b.BehaviorFunc(ctx)));
            // return b => b.Selelct(f);
        }

        public static Func<Behavior<T1>, Behavior<T2>, Behavior<R>> Lift<T1, T2, R>(Func<T1, T2, R> f)
        {
            return (b1, b2) => Create(ctx =>
              f(b1.BehaviorFunc(ctx), b2.BehaviorFunc(ctx)));
        }

        public static Func<Behavior<T1>, Behavior<T2>, Behavior<T3>, Behavior<R>> Lift<T1, T2, T3, R>(Func<T1, T2, T3, R> f)
        {
            return (b1, b2, b3) =>
                Create(ctx => f(b1.BehaviorFunc(ctx), b2.BehaviorFunc(ctx), b3.BehaviorFunc(ctx)));
        }
    }


    public static class Time
    {
        // Behavior that represents the current time
        public static Behavior<float> Current
        {
            get { return Behavior.Create(ctx => ctx.Time); }
        }
        // A value oscillating between 1 and -1
        public static Behavior<float> CircularAnim
        {
            get
            {
                return Behavior.Create(ctx => (float)Math.Sin(ctx.Time * Math.PI));
            }
        }
        // Create constant behavior from a value
        public static Behavior<T> Forever<T>(T v)
        {
            return Behavior.Create(ctx => v);
        }
        // Extension method for floats only
        public static Behavior<float> Forever(this float v)
        {
            return Behavior.Create(ctx => v);
        }
        public static Behavior<T> Faster<T>(this Behavior<T> v, float speed)
        {
            return Behavior.Create(ctx => v.BehaviorFunc(new BehaviorContext(ctx.Time * speed)));
        }
        public static Behavior<T> Wait<T>(this Behavior<T> v, float delay)
        {
            return Behavior.Create(ctx => v.BehaviorFunc(new BehaviorContext(ctx.Time + delay)));
        }
    }

    public interface IDrawing
    {
        void Draw(Graphics gr);
    }

    class Drawing : IDrawing
    {
        public Drawing(Action<Graphics> f)
        {
            this.f = f;
        }
        private readonly Action<Graphics> f;
        public Action<Graphics> DrawFunc { get { return f; } }

        public void Draw(Graphics gr)
        {
            DrawFunc(gr);
        }
    }

    public static class Drawings
    {
        public static IDrawing Circle(Brush brush, float size)
        {
            return new Drawing(gr =>
                      gr.FillEllipse(brush, -size / 2.0f, -size / 2.0f, size, size)
                  );
        }
        public static IDrawing Translate(this IDrawing img, float x, float y)
        {
            return new Drawing(g =>
            {
                g.TranslateTransform(x, y);
                img.Draw(g);
                g.TranslateTransform(-x, -y);
            }
                  );
        }

        public static IDrawing Compose(this IDrawing img1, IDrawing img2)
        {
            return new Drawing(g =>
            {
                img1.Draw(g);
                img2.Draw(g);
            }
            );
        }
    }

    public static class Anims
    {
        // Lifted version of drawing operations
        public static Behavior<IDrawing> Compose(this Behavior<IDrawing> anim1, Behavior<IDrawing> anim2)
        {
            return Behavior.Lift<IDrawing, IDrawing, IDrawing>(Drawings.Compose)(anim1, anim2);
        }

        public static Behavior<IDrawing> Cirle(Behavior<Brush> brush, Behavior<float> size)
        {
            return Behavior.Lift<Brush, float, IDrawing>(Drawings.Circle)(brush, size);
        }

        public static Behavior<IDrawing> Translate(this Behavior<IDrawing> drawing, Behavior<float> x, Behavior<float> y)
        {
            return Behavior.Lift<IDrawing, float, float, IDrawing>(Drawings.Translate)(drawing, x, y);
        }
    }
}
