module DiscUnion

type Shape  =
  | Square  of int
  | Rectangle of float * float
  | Circle of float


let sq = Square 7
let rect = Rectangle (2.2, 3.3)
let cir = Circle 3.4

let getArea shape =
  match shape with
  | Square side -> float(side * side)
  | Rectangle(w,h) -> w * h
  | Circle r -> System.Math.PI * r * r
