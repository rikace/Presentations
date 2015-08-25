open System

#r "../Lib/OpenTK.dll"
#r "../Lib/OpenTK.GLControl.dll"
#load "Common/functional3d.fs"


module ``DSL Graphic`` =


    open Functional3D
    open System.Drawing

    // ------------------------------------------------------------------

    Fun.color Color.Yellow Fun.cylinder

    ( Fun.color Color.Yellow Fun.cylinder ) $
    ( Fun.cone
      |> Fun.color Color.Red 
      |> Fun.translate (0.0, 0.0, -1.0) )

    // ------------------------------------------------------------------

    let tower x z = 
      (Fun.cylinder
         |> Fun.scale (1.0, 1.0, 3.0) 
         |> Fun.translate (0.0, 0.0, 1.0)
         |> Fun.color Color.DarkGoldenrod ) $ 
      (Fun.cone 
         |> Fun.scale (1.3, 1.3, 1.3) 
         |> Fun.translate (0.0, 0.0, -1.0)
         |> Fun.color Color.Red )
      |> Fun.rotate (90.0, 0.0, 0.0)
      |> Fun.translate (x, 0.5, z)

    // Create one tower
    tower 0.0 0.0

    // Now we can easily compose towers!
    tower -2.0 0.0 $ tower 2.0 0.0
                                                                                                                                
    // ------------------------------------------------------------------

    let sizedCube height = 
      Fun.cube 
      |> Fun.scale (0.5, height, 1.0) 
      |> Fun.translate (-0.5, height/2.0 - 1.0, 0.0)

    let twoCubes =
      sizedCube 0.8 $ (sizedCube 1.0 |> Fun.translate (0.5, 0.0, 0.0))

    let block = 
      [ for offset in -4.0 .. +4.0 ->
          twoCubes |> Fun.translate (offset, 0.0, 0.0) ]
      |> Seq.reduce ($)
      |> Fun.scale (0.5, 2.0, 0.3)
      |> Fun.color Color.DarkGray
  
    // ------------------------------------------------------------------

    let wall offs rotate = 
      let rotationArg = if rotate then (0.0, 90.0, 0.0) else (0.0, 0.0, 0.0)
      let translationArg = if rotate then (offs, 0.0, 0.0) else (0.0, 0.0, offs)
      block |> Fun.rotate rotationArg |> Fun.translate translationArg

    tower -2.0 -2.0 $ tower 2.0 -2.0 $ 
      tower -2.0 2.0 $ tower 2.0 2.0 $
      wall -2.0 true $ wall 2.0 true $
      wall -2.0 false $ wall 2.0 false
















    // ------------------------------------------------------------------
    // Recursion 
    // ------------------------------------------------------------------

    let pattern = 
      [| [| [| 1; 1; 1; |]; [| 1; 0; 1 |]; [| 1; 1; 1 |] |]
         [| [| 1; 0; 1; |]; [| 0; 0; 0 |]; [| 1; 0; 1 |] |]
         [| [| 1; 1; 1; |]; [| 1; 0; 1 |]; [| 1; 1; 1 |] |] |]
      |> Array3D.fromCube

    let rec generate depth = 
      [ for x in -1 .. 1 do
        for y in -1 .. 1 do
        for z in -1 .. 1 do
          if pattern.[x, y, z] = 1 then 
            let size = 3.0 ** float depth
            let ofs = float x * size, float y * size, float z * size
            let sub = if depth = 0 then Fun.cube
                      else generate (depth - 1) 
            yield Fun.translate ofs sub ]
      |> List.reduce ($)
      |> Fun.color Color.ForestGreen

    // Generate fractal with various level of detail
  
    Fun.setDistance(-20.0)

    generate 0
    generate 1

    Fun.setDistance(-60.0)
    generate 2

    // ------------------------------------------------------------------
    // Trees are an example of recursive structure
    // ------------------------------------------------------------------

    let random = System.Random()

    let noise k x =
      x + (k * x * (random.NextDouble() - 0.5))

    let color() = 
      [| Color.Red; Color.Orange; 
         Color.Yellow |].[random.Next 3]

    let trunk (width,length) = 
      Fun.cylinder
      |> Fun.translate (0.0,0.0,0.5) |> Fun.scale (width,width,length)  
        
    let fruit (size) = 
      Fun.sphere
      |> Fun.color (color()) |> Fun.scale (size,size,size)

    let example = trunk (1.0,5.0) $ fruit 2.0


    // Recursive tree
    let rec tree trunkLength trunkWidth w n = 
      let moveToEndOfTrunk = Fun.translate (0.0,0.0,trunkLength)
      if n <= 1 then
        trunk (trunkWidth,trunkLength) $  // branch and end with
        (fruit (3.0 * trunkWidth) |> moveToEndOfTrunk)  // fruit
      else 
        // generate branch
        let branch angleX angleY = 
          let branchLength = trunkLength * 0.92 |> noise 0.2  // reduce length
          let branchWidth  = trunkWidth  * 0.65 |> noise 0.2  // reduce width
          tree branchLength branchWidth w (n-1) 
          |> Fun.rotate (angleX,angleY,0.0) |> moveToEndOfTrunk
      
        trunk (trunkWidth,trunkLength)  // branch and follow by several
          $ branch  w  0.0              // smaller branches with rotation +/- w
          $ branch -w  0.0
          $ branch 0.0   w
          $ branch 0.0  -w

    let plant = 
      tree 4.0(*long*) 0.8(*wide*) 40.0(*angle*) 4(*levels*)
      |> Fun.rotate (90.0, 180.0, 90.0)
      |> Fun.translate (0.0, -6.0, 0.0)


    Fun.resetRotation()


module ``DSL dependent types`` =

    type PersonalName = 
        {
            FirstName: string;
            LastName: string;
        }

        //Should these constraints be part of the domain model?

        (*
        void SaveToDatabase(PersonalName personalName)
        { 
           var first = personalName.First;
           if (first.Length > 50)
           {    
                // ensure string is not too long
                first = first.Substring(0,50);
           }
   
           //save to database
        } *)


        //  If the string is too long at this point, what should you do? 
        //  Silently truncate it? Throw an exception?

        //  A better answer is to avoid the problem altogether if you can. 
        //  By the time the string gets to the database layer it is too late 
                //  the database layer should not be making these kinds of decisions.

        // The problem should be deal with when the string was first created, not when it is used. 
        
    module String100 = 
        type T = String100 of string

        let create (s:string) = 
            if not(String.IsNullOrEmpty(s)) && s.Length <= 100 
            then Some (String100 s) 
            else None
        let apply f (String100 s) = f s
        let value s = apply id s



    module String50 = 
        type T = String50 of string
        let create (s:string) = 
            if not(String.IsNullOrEmpty(s)) && s.Length <= 50 
            then Some (String50 s) 
            else None
        let apply f (String50 s) = f s
        let value s = apply id s 



    module String2 = 
        type T = String2 of string
        let create (s:string) = 
            if not(String.IsNullOrEmpty(s)) && s.Length <= 2 
            then Some (String2 s) 
            else None
        let apply f (String2 s) = f s
        let value s = apply id s


        //  One problem is that we have a lot of duplicated code. 
        //  In practice a typical domain only has a few dozen string types, 
        //  so there won't be that much wasted code. But still, we can probably do better.

        //  Another more serious problem is that comparisons become harder. 
        //  A String50 is a different type from a String100 so that they cannot be compared directly.

    let s50 = String50.create "John"
    let s100 = String100.create "Smith"

    let s50' = s50.Value
    let s100' = s100.Value

    let areEqual = (s50' = s100')  // compiler error


        //  Refactoring 
    module WrappedString = 

        /// An interface that all wrapped strings support
        type IWrappedString = 
            abstract Value : string

        /// Create a wrapped value option
        /// 1) canonicalize the input first
        /// 2) If the validation succeeds, return Some of the given constructor
        /// 3) If the validation fails, return None
        let create canonicalize isValid ctor (s:string) = 
            if String.IsNullOrEmpty(s)
            then None
            else
                let s' = canonicalize s
                if isValid s'
                then Some (ctor s') 
                else None

        /// Apply the given function to the wrapped value
        let apply f (s:IWrappedString) = 
            s.Value |> f 

        /// Get the wrapped value
        let value s = apply id s

        /// Equality test
        let equals left right = 
            (value left) = (value right)
        
        /// Comparison
        let compareTo left right = 
            (value left).CompareTo (value right)


        /// Canonicalizes a string before construction
        /// * converts all whitespace to a space char
        /// * trims both ends
        let singleLineTrimmed s =
            System.Text.RegularExpressions.Regex.Replace(s,"\s"," ").Trim()

        /// A validation function based on length
        let lengthValidator len (s:string) =
            s.Length <= len 

        /// A string of length 100
        type String100 = String100 of string with
            interface IWrappedString with
                member this.Value = let (String100 s) = this in s

        /// A constructor for strings of length 100
        let string100 = create singleLineTrimmed (lengthValidator 100) String100 

        /// Converts a wrapped string to a string of length 100
        let convertTo100 s = apply string100 s

        /// A string of length 50
        type String50 = String50 of string with
            interface IWrappedString with
                member this.Value = let (String50 s) = this in s

        /// A constructor for strings of length 50
        let string50 = create singleLineTrimmed (lengthValidator 50)  String50

        /// Converts a wrapped string to a string of length 50
        let convertTo50 s = apply string50 s


            /// A multiline text of length 1000
        type Text1000 = Text1000 of string with
            interface IWrappedString with
                member this.Value = let (Text1000 s) = this in s

        /// A constructor for multiline strings of length 1000
        let text1000 = create id (lengthValidator 1000) Text1000 



    module PersonalNameConstraint = 
        open WrappedString

        type T = 
            {
            FirstName: String50;
            LastName: String100;
            }

        /// create a new value
        let create first last = 
            match (string50 first),(string100 last) with
            | Some f, Some l ->
                Some {
                    FirstName = f;
                    LastName = l;
                    }
            | _ -> None



    let name = PersonalNameConstraint.create "John" "Smith"



    // ========================================
    // Email address (not application specific)
    // ========================================

    module EmailAddress = 

        type T = EmailAddress of string with 
            interface WrappedString.IWrappedString with
                member this.Value = let (EmailAddress s) = this in s

        let create = 
            let canonicalize = WrappedString.singleLineTrimmed 
            let isValid s = 
                (WrappedString.lengthValidator 100 s) &&
                System.Text.RegularExpressions.Regex.IsMatch(s,@"^\S+@\S+\.\S+$") 
            WrappedString.create canonicalize isValid EmailAddress

        /// Converts any wrapped string to an EmailAddress
        let convert s = WrappedString.apply create s

    // ========================================
    // ZipCode (not application specific)
    // ========================================

    module ZipCode = 

        type T = ZipCode of string with
            interface WrappedString.IWrappedString with
                member this.Value = let (ZipCode s) = this in s

        let create = 
            let canonicalize = WrappedString.singleLineTrimmed 
            let isValid s = 
                System.Text.RegularExpressions.Regex.IsMatch(s,@"^\d{5}$") 
            WrappedString.create canonicalize isValid ZipCode

        /// Converts any wrapped string to a ZipCode
        let convert s = WrappedString.apply create s

    // ========================================
    // StateCode (not application specific)
    // ========================================

    module StateCode = 

        type T = StateCode  of string with
            interface WrappedString.IWrappedString with
                member this.Value = let (StateCode  s) = this in s

        let create = 
            let canonicalize = WrappedString.singleLineTrimmed 
            let stateCodes = ["AZ";"CA";"NY"] //etc
            let isValid s = 
                stateCodes |> List.exists ((=) s)

            WrappedString.create canonicalize isValid StateCode

        /// Converts any wrapped string to a StateCode
        let convert s = WrappedString.apply create s

    // ========================================
    // PostalAddress (not application specific)
    // ========================================

    module PostalAddress = 

        type USPostalAddress = 
            {
            Address1: WrappedString.String50;
            Address2: WrappedString.String50;
            City: WrappedString.String50;
            State: StateCode.T;
            Zip: ZipCode.T;
            }

        type UKPostalAddress = 
            {
            Address1: WrappedString.String50;
            Address2: WrappedString.String50;
            Town: WrappedString.String50;
            PostCode: WrappedString.String50;   // todo
            }

        type GenericPostalAddress = 
            {
            Address1: WrappedString.String50;
            Address2: WrappedString.String50;
            Address3: WrappedString.String50;
            Address4: WrappedString.String50;
            Address5: WrappedString.String50;
            }

        type T = 
            | USPostalAddress of USPostalAddress 
            | UKPostalAddress of UKPostalAddress 
            | GenericPostalAddress of GenericPostalAddress 