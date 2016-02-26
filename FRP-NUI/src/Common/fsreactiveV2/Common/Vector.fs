#light

namespace Common

 module Vector =

    open System 

    type  Vector = Vector of (float * float)
        with 
            static member (+) ((Vector (xa,ya)), (Vector (xb, yb))) = Vector ((xa+xb), (ya+yb))
            static member (-) ((Vector (xa,ya)), (Vector (xb, yb))) = Vector ((xa-xb), (ya-yb))
            static member (*) ((Vector (xa,ya)), k) = Vector (k*xa, k*ya)
            static member (/) ((v:Vector), k) = v * (1.0/k)
            static member dot (Vector (xa,ya)) (Vector (xb,yb)) = xa*xb+ya*yb
            static member neg (Vector (xa,ya)) = Vector (-xa, -ya)
            static member length (Vector (xa,ya)) = Math.Sqrt(xa*xa+ya*ya)
            static member norm (v:Vector) = v / (Vector.length v)
            static member rot (Vector (x,y)) angle    = let cosa = Math.Cos angle
                                                        let sina = Math.Sin angle
                                                        Vector ((x * cosa - y * sina), (x * sina + y * cosa))
            static member unit = Vector(1.0, 0.0)
            static member zero = Vector(0.0, 0.0)

            