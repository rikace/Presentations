namespace DataModel
    open System.ComponentModel.DataAnnotations
    open System.Data.Entity
    open System

    [<Interface>]
    type ICourse =
        abstract member CourseID : int with get,set
        abstract member Name : string with get,set

    [<Interface>]
    type ICourseLength =
        abstract member Hours : int with get,set

    [<Interface>]
    type IStudentAge =
        abstract member Age : int Nullable with get,set
        abstract member Score : int with get,set
        abstract member Name : string with get,set

    [<Interface>]
    type IStudentName =
        abstract member FirstName : string with get,set
        abstract member LastName : string with get,set

    [<Interface>]
    type IStudent =
        inherit IStudentAge
        inherit IStudentName

    type public Student() =
        [<Key>]
        member val StudentID = Unchecked.defaultof<int> with get,set
        member val FirstName = Unchecked.defaultof<string> with get,set
        member val LastName = Unchecked.defaultof<string> with get,set
        member val Name = Unchecked.defaultof<string> with get,set
        member val Age = Unchecked.defaultof<int Nullable> with get,set
        member val Score = Unchecked.defaultof<int> with get,set

        interface IStudentAge with
            member val Age = Unchecked.defaultof<int Nullable> with get,set
            member val Score = Unchecked.defaultof<int> with get,set
            member val Name = Unchecked.defaultof<string> with get,set

        interface IStudentName with
            member val FirstName = Unchecked.defaultof<string> with get,set
            member val LastName = Unchecked.defaultof<string> with get,set


    type Course() =
        [<Key>]
        member val CourseID = Unchecked.defaultof<int> with get,set
        member val Name = Unchecked.defaultof<string> with get,set
        member val Hours = Unchecked.defaultof<int> with get,set

        interface ICourseLength with
            member val Hours = Unchecked.defaultof<int> with get,set

        interface ICourse with
            member val CourseID = Unchecked.defaultof<int> with get,set
            member val Name = Unchecked.defaultof<string> with get,set


//    type SimpleDbContext() =
//        inherit DbContext("name=SimpleDbConnection")
//
//        [<DefaultValue>]
//        val mutable students : DbSet<Student>
//        member self.Students with get() = self.students
//                             and set value = self.students <- value
//
//        [<DefaultValue>]
//        val mutable courses : DbSet<Course>
//        member self.Courses with get() = self.courses
//                             and set value = self.courses <- value

     type SimpleDbContext(connString:string) =
            inherit DbContext(connString)

            [<DefaultValue>]
            val mutable students : DbSet<Student>
            member self.Students with get() = self.students
                                 and set value = self.students <- value

            [<DefaultValue>]
            val mutable courses : DbSet<Course>
            member self.Courses with get() = self.courses
                                 and set value = self.courses <- value
