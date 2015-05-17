// it is easy to create DDD types in FP 
// here are an example in F# of a simple “customer” domain.

type PersonalName = {FirstName:string; LastName:string}

// Addresses
type StreetAddress = {Line1:string; Line2:string; Line3:string }

type ZipCode =  ZipCode of string   
type StateAbbrev =  StateAbbrev of string

type ZipAndState =  {State:StateAbbrev; Zip:ZipCode }
type USAddress = {Street:StreetAddress; Region:ZipAndState}

type UKPostCode =  PostCode of string
type UKAddress = {Street:StreetAddress; Region:UKPostCode}

type InternationalAddress = {
    Street:StreetAddress; Region:string; CountryName:string}

// choice type  -- must be one of these three specific types
type Address = USAddress | UKAddress | InternationalAddress

// Email
type Email = Email of string

// Phone
type CountryPrefix = Prefix of int
type Phone = {CountryPrefix:CountryPrefix; LocalNumber:string}

type Contact = 
    {
        PersonalName: PersonalName;
        // "option" means it might be missing
        Address: Address option;
        Email: Email option;
        Phone: Phone option;
    }

// Put it all together into a CustomerAccount type
type CustomerAccountId  = AccountId of string
type CustomerType  = Prospect | Active | Inactive

// override equality and deny comparison
[<CustomEquality; NoComparison>]
type CustomerAccount = 
    {
        CustomerAccountId: CustomerAccountId;
        CustomerType: CustomerType;
        ContactInfo: Contact;
    }

    override this.Equals(other) =
        match other with
        | :? CustomerAccount as otherCust -> 
          (this.CustomerAccountId = otherCust.CustomerAccountId)
        | _ -> false

    override this.GetHashCode() = hash this.CustomerAccountId 


(* 
This code fragment contains 17 type definitions in just a few lines, but with minimal complexity.
This is a simplified version with just the basic types
In a real system, constraints and other methods would be added. 
*)
