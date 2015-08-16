open System

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