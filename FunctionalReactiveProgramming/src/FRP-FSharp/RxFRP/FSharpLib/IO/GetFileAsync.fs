namespace Easj360FSharp
//
//open System.IO
//open System
//open System.Runtime.ConstrainedExecution
//open System.Runtime.InteropServices
//open System.Security.Permissions
//open Microsoft.Win32.SafeHandles
//
//
//module FileNative =
// 
//
//  [<ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)>]
//  [<DllImport("kernel32.dll")>]
//  extern bool FindClose(nativeint handle)
//
//  type SafeFindHandle() =
//    inherit SafeHandleZeroOrMinusOneIsInvalid(true)
//    override this.ReleaseHandle() = FindClose(this.handle)
//
//  [<Serializable; Struct; StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto);BestFitMapping(false)>]
//  type WIN32_FIND_DATA =
//    val mutable dwFileAttributes:FileAttributes
//    val mutable ftCreationTime_dwLowDateTime:int
//    val mutable ftCreationTime_dwHighDateTime:int
//    val mutable ftLastAccessTime_dwLowDateTime:int
//    val mutable ftLastAccessTime_dwHighDateTime:int
//    val mutable ftLastWriteTime_dwLowDateTime:int
//    val mutable ftLastWriteTime_dwHighDateTime:int
//    val mutable nFileSizeHigh:int
//    val mutable nFileSizeLow:int
//    val mutable dwReserved0:int
//    val mutable dwReserved1:int
//    [<MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)>]
//    val mutable cFileName:string
//    [<MarshalAs(UnmanagedType.ByValTStr, SizeConst = 14)>]
//    val mutable cAlternateFileName:string
//
//  [<DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)>]
//  extern SafeFindHandle FindFirstFile(string fileName, [<In; Out>] WIN32_FIND_DATA& data)
//
//  [<DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)>]
//  extern bool FindNextFile(SafeFindHandle hndFindFile, [<In; Out; MarshalAs(UnmanagedType.LPStruct)>] WIN32_FIND_DATA& lpFindFileData)
//
//  let rec yieldFiles (path:string) (data:WIN32_FIND_DATA ref) (handle:SafeFindHandle) = seq {
//    let data' = !data
//    if (data'.dwFileAttributes &&& FileAttributes.Directory) <> (enum 0) then
//      if data'.cFileName <> "." && data'.cFileName <> ".." then
//        let subdirectory = Path.Combine(path, data'.cFileName)
//        yield! getFiles subdirectory
//      else
//        yield Path.Combine(path, data'.cFileName)
//        let data''= ref (new WIN32_FIND_DATA())
//        if FindNextFile(handle, !data'') then
//          yield! yieldFiles path data'' handle }
//
//  and getFiles (path : string) =
//    seq {
//    let findData = ref (new WIN32_FIND_DATA())
//    use findHandle = FindFirstFile(path + @"\*", !findData)
//    let findData' = findData
//    if not findHandle.IsInvalid then
//      yield! yieldFiles path findData' findHandle }
//
//
////// C# original
////public IEnumerable<string> GetFiles(string directory) {
////    var findData = new NativeMethods.WIN32_FIND_DATA();
////    using(var findHandle = NativeMethods.FindFirstFile(directory + @"\*", findData)) {
////        if (!findHandle.IsInvalid) {
////            do {
////                if ((findData.dwFileAttributes & FileAttributes.Directory) != 0) {
////                    if (findData.cFileName != "." && findData.cFileName != "..") {
////                        var subdirectory = Path.Combine(directory, findData.cFileName);
////                        foreach (var file in GetFiles(subdirectory))
////                            yield return file;
////                    }
////                } else {
////                    var path = Path.Combine(directory, findData.cFileName);
////                    yield return Path.Combine(directory, findData.cFileName);
////                        
////            } while (NativeMethods.FindNextFile(findHandle, findData));
////        }
////    }
////}