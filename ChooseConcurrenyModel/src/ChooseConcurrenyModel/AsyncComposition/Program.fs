
open CommonHelpers

//#nowarn "1189"
/// Given a readonly state, produces a value
type Reader<'TState, 'T> = 'TState -> 'T
/// Produces a value together with additional state
type Writer<'TState, 'T> = 'TState * 'T
/// Given state, produces new state & a value
type State<'TState, 'T>  = 'TState -> 'TState * 'T
/// Represents an update monad - given a state, produce 
/// value and an update that can be applied to the state
type UpdateMonad<'TState, 'TUpdate, 'T> = 
  UM of ('TState -> 'TUpdate * 'T)

/// Returns the value of 'Unit' property on the ^S type
let inline unit< ^S when ^S : 
    (static member Unit : ^S)> () : ^S =
  (^S : (static member Unit : ^S) ()) 

/// Invokes Combine operation on a pair of ^S values
let inline (++)< ^S when ^S : 
    (static member Combine : ^S * ^S -> ^S )> a b : ^S =
  (^S : (static member Combine : ^S * ^S -> ^S) (a, b)) 

/// Invokes Apply operation on state and update ^S * ^U
let inline apply< ^S, ^U when ^U : 
    (static member Apply : ^S * ^U -> ^S )> s a : ^S =
  (^U : (static member Apply : ^S * ^U -> ^S) (s, a)) 

type UpdateBuilder() = 
  /// Returns the specified value, together
  /// with empty update obtained using 'unit'
  member inline x.Return(v) : UpdateMonad<'S, 'U, 'T> = 
    UM (fun s -> (unit(),v))

  /// Compose two update monad computations
  member inline x.Bind(UM u1, f:'T -> UpdateMonad<'S, 'U, 'R>) =  
    UM (fun s -> 
      // Run the first computation to get first update
      // 'u1', then run 'f' to get second computation
      let (u1, x) = u1 s
      let (UM u2) = f x
      // Apply 'u1' to original state & run second computation
      // then return result with combined state updates
      let (u2, y) = u2 (apply s u1)
      (u1 ++ u2, y))

/// Instance of the computation builder
/// that defines the update { .. } block
let update = UpdateBuilder()

/// The state of the reader is 'int'
type ReaderState = int
/// Trivial monoid of updates 
type ReaderUpdate = 
  | NoUpdate
  static member Unit = NoUpdate
  static member Combine(NoUpdate, NoUpdate) = NoUpdate
  static member Apply(s, NoUpdate) = s

/// Read the current state (int) and return it as 'int'
let read = UM (fun (s:ReaderState) -> 
  (NoUpdate, s))
/// Run computation and return the result 
let readRun (s:ReaderState) (UM f) = f s |> snd

/// Returns state + 1
let demo1 = update { 
  let! v = read
  return v + 2 }
/// Returns the result of demo1 + 1
let demo2 = update { 
  let! v = demo1
  return v + 1 }

// Run it with state 40 
demo2 |> readRun 40

/// Writer monad has no readable state
type WriterState = NoState

/// Updates of writer monad form a list
type WriterUpdate<'TLog> = 
  | Log of list<'TLog>
  /// Returns the empty log (monoid unit)
  static member Unit = Log []
  /// Combines two logs (operation of the monoid)
  static member Combine(Log a, Log b) = Log(List.append a b)
  /// Applying updates to state does not affect the state
  static member Apply(NoState, _) = NoState

/// Writes the specified value to the log 
let write v = UM (fun s -> (Log [v], ()))
/// Runs a "writer monad computation" and returns 
/// the log, together with the final result
let writeRun (UM f) = let (Log l, v) = f NoState in l, v

/// Writes '20' to the log and returns "world"
let demo3 = update {
  do! write 20
  return "world" }
/// Calls 'demo3' and then writes 10 to the log
let demo4 = update {
  let! w = demo3
  do! write 10
  return "Hello " + w }

/// Returns "Hello world" with 20 and 10 in the log
demo4 |> writeRun


/// Extends UpdateBuilder to support additional syntax
type UpdateBuilder with
  /// Represents monadic computation that returns unit
  /// (e.g. we can now omit 'else' branch in 'if' computation)
  member inline x.Zero() = x.Return(())

  /// Delays a computation with (uncontrolled) side effects
  member inline x.Delay(f) = x.Bind(x.Zero(), f)

  /// Sequential composition of two computations where the
  /// first one has no result (returns a unit value)
  member inline x.Combine(c1, c2) = x.Bind(c1, fun () -> c2)

  /// Enable the 'return!' keyword to return another computation
  member inline x.ReturnFrom(m : UpdateMonad<'S, 'P, 'T>) = m

  /// Ensure that resource 'r' is disposed of at the end of the
  /// computation specified by the function 'f'
  member inline x.Using(r,f) = UM(fun s -> 
    use rr = r in let (UM g) = f rr in g s)

  /// Support 'for' loop - runs body 'f' for each element in 'sq'
  member inline x.For(sq:seq<'V>, f:'V -> UpdateMonad<'S, 'P, unit>) = 
    let rec loop (en:System.Collections.Generic.IEnumerator<_>) = 
      if en.MoveNext() then x.Bind(f en.Current, fun _ -> loop en)
      else x.Zero()
    x.Using(sq.GetEnumerator(), loop)

  /// Supports 'while' loop - run body 'f' until condition 't' holds
  member inline x.While(t, f:unit -> UpdateMonad<'S, 'P, unit>) =
    let rec loop () = 
      if t() then x.Bind(f(), loop)
      else x.Zero()
    loop()

(**
You can find more details about these operations in the [F# Computation Zoo paper][zoopaper]
or in the [F# language specification][fsspec]. In fact, the definitions mostly follow
the samples from the F# specification. It is worth noting that all the members are
marked as `inline`, which allows us to use _static member constrains_ and to write
code that will work for any update monad (as defined by a pair of _update_ and 
_state_ types).

Let's look at a trivial example using the writer computation:
*)
/// Logs numbers from 1 to 10
let logNumbers = update {
  for i in 1 .. 10 do 
    do! write i }
(**
As expected, when we run the computation using `writeRun`, the result is a tuple containing
a list with numbers from 1 to 10 and a unit value. The computation does not explicitly 
return and so the `Zero` member is automatically used.

Implementing the state monad
----------------------------

Interestingly, the standard state monad is _not_ a special case of update monads. However, we
can define a computation that implements the same functionality - a computation with state
that we can read and write.

### States and updates

In this final example, both the type representing _state_ and the type representing
_update_ will have a useful role. We make both of the types generic over the value they
carry. State is simply a wrapper containing the value (current state). Update can be of
two kinds - we have an empty update (do nothing) and an update to set the state:

*)
/// Wraps a state of type 'T
type StateState<'T> = State of 'T

/// Represents updates on state of type 'T
type StateUpdate<'T> = 
  | Set of 'T | SetNop
  /// Empty update - do not change the state
  static member Unit = SetNop
  /// Combine updates - return the latest (rightmost) 'Set' update
  static member Combine(a, b) = 
    match a, b with 
    | SetNop, v | v, SetNop -> v 
    | Set a, Set b -> Set b
  /// Apply update to a state - the 'Set' update changes the state
  static member Apply(s, p) = 
    match p with SetNop -> s | Set s -> State s
(**
This definition is a bit more interesting than the previous two, because there is some
interaction between the _states_ and _updates_. In particular, when the update is `Set v`
(we want to replace the current state with a new one), the `Apply` member returns a new
state instead of the original. For the `Unit` member, we need an update `SetNop` which 
simply means that we want to keep the original state (and so `Apply` just returns the
original value in this case).

Another notable thing is the `Combine` operation - it takes two updates (which may be 
either empty updates or set updates) and produces a single one. If you read a composition
`a1 ++ a2 ++ .. ++ an` as a sequence of state updates (either `Set` or `SetNop`), then the 
`Combine` operation returns the last `Set` update in the sequence (or `SetNop` if there are
no `Set` updates). In other words, it builds an update that sets the last state that was
set during the whole sequence.

### State monad primitives

Now that we have the type definitions, it is quite easy to add the usual primitives:
*)
/// Set the state to the specified value
let set s = UM (fun _ -> (Set s,()))
/// Get the current state 
let get = UM (fun (State s) -> (SetNop, s))
/// Run a computation using a specified initial state
let setRun s (UM f) = f (State s) |> snd
(**
The `set` operation is a bit different than the usual one for state monad. It ignores the
state and it builds an _update_ that tells the computation to set the new state. 
The `get` operation reads the state and returns it - but as it does not intend to change it,
it returns `SetNop` as the update.

### Sample stateful computation

If you made it this far in the article, you can already expect how the example will look!
We'll again use the `update { .. }` computation. This time, we define a computation
`demo5` that increments the state and call it from a loop in `demo6`:
*)
/// Increments the state by one
let demo5 = update { 
  let! v = get
  do! set (v + 1) }
/// Call 'demo5' repeatedly in a loop
/// and then return the final state
let demo6 = update {
  for i in 1 .. 10 do 
    do! demo5
  return! get }
// Run the sample with initial state 0
demo6 |> setRun 0
(**
Running the code yields 10 as expected - we start with zero and then increment the
state ten times. Since we extended the definition of the `UpdateBuilder` (in the 
previous section), we now got a few nice things for free - we can use the `for` loop
and write computations (like `demo5`) without explicit `return` if they just need to
modify the state.

Conclusions
-----------

People coming to F# from the Haskell background often dislike the fact that
F# does not let you write code polymorphic over monads and that computation
expressions always explicitly state the type of computations such as 
`async { .. }`. I think there are good reasons for this and tried to explain some
of them in [a recent blog post and PADL paper][zoo].

As a result, using reader, writer and state monads in F# was always a bit 
cumbersome. In this blog post, I looked at an F# implementation of the recent
idea called _update monads_ (see [the original paper (PDF)][um]), which unifies
the three state-related monads into a single type. This works very nicely with F#
- we can define just a single computation builder for all state-related computations
and then define a concrete state-related monad by defining two simple types.
I used the approach to define a reader monad, writer monad useful for logging and
a state monad (that keeps a state and allows changing it).

I guess that making update monads part of standard library and standard programming
style in Haskell will be tricky because of historical reasons. However, for F#
libraries that try to make purely functional programming easier, I think that 
update monads are the way to go.


  [zoo]: http://tomasp.net/blog/2013/computation-zoo-padl
  [zoopaper]: http://tomasp.net/academic/papers/computation-zoo/
  [um]: http://cs.ioc.ee/~tarmo/papers/types13.pdf
  [smc]: http://msdn.microsoft.com/en-us/library/dd233203.aspx
  [fsspec]: http://fsharp.org/about/index.html#specification
*)






[<EntryPoint>]
let main argv = 

    BenchPerformance.Time("Load Data Async Paralall and Process", fun () ->
        let matches = AsyncCompositionModule.loadDataAsyncInParalallAndProcess()
        for m in matches do 
            printf "%s\t" m )

    System.Console.ReadKey() |> ignore

    0 // return an integer exit code
