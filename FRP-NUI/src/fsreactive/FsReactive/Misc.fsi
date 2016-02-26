#light


module Misc = begin
  val catOption : 'a option list -> 'a list
  val memoize : ('a -> 'b) -> ('a -> 'b)
end
