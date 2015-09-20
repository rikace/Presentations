namespace Utility

//#light
//
//#r "Microsoft.Office.Interop.Excel.dll"
//#r "office.dll"

open System
open System.IO
open System.Reflection
open Microsoft.Office.Interop.Excel

module srcModuleExcel = 

    let app = ApplicationClass(Visible = true)

    let sheet = app.Workbooks
                   .Add()
                   .Worksheets.[1] :?> _Worksheet

    let setCellText (x : int) (y : int) (text : string) = 
        let range = sprintf "%c%d" (char (x + int 'A')) (y+1)
        sheet.Range(range).Value(Missing.Value) <- text


    let printCsvToExcel rowIdx (csvText : string) =
        csvText.Split([| ',' |])
        |> Array.iteri (fun partIdx partText -> setCellText partIdx rowIdx partText)
    

    let rec filesUnderFolder basePath = 
        seq {
            yield! Directory.GetFiles(basePath)
            for subFolder in Directory.GetDirectories(basePath) do
                yield! filesUnderFolder subFolder 
        }

    let doStuff3() =

        // Print header
        printCsvToExcel 0 "Directory, Filename, Size, Creation Time"

        // Print rows
        filesUnderFolder (Environment.GetFolderPath(Environment.SpecialFolder.MyPictures))
        |> Seq.map (fun filename -> new FileInfo(filename))
        |> Seq.map (fun fileInfo -> sprintf "%s, %s, %d, %s" 
                                        fileInfo.DirectoryName 
                                        fileInfo.Name 
                                        fileInfo.Length 
                                        (fileInfo.CreationTime.ToShortDateString()))
        |> Seq.iteri (fun idx str -> printCsvToExcel (idx + 1) str)


    //////////////////////////

    // Run Excel as a visible application
    let app2 = new ApplicationClass(Visible = true) 
    // Create new file using the default template
    let workbook = app2.Workbooks.Add(XlWBATemplate.xlWBATWorksheet) 
    // Get the first worksheet
    let worksheet = (workbook.Worksheets.[1] :?> _Worksheet) 

    // Write values to the worksheet
    worksheet.Range("C2").Value2 <- "1990"
    worksheet.Range("C2", "E2").Value2 <- [| "1990"; "2000"; "2005" |]


    let stats = [| ("MD", 25); ("VA", 106); ("WV", 175) |]

    let statsArr = stats |> Seq.toArray
    // Get names of regions as 2D array
    let names = statsArr |> Array.map (fun x -> fst x)
    let namesVert = Array2D.init names.Length 1 (fun i _ -> names.[i])
    // Initialize 2D array with the data
    let tableArr = Array2D.init statsArr.Length 3 (fun x y -> 
      // Read value for a year 'y' from the i-th region
      let _, values = statsArr.[x]
      // Display millions of square kilometers
      float(values) * 1.5)

    // Write the data to the worksheet
    let slen = string(statsArr.Length + 2)
    worksheet.Range("B3", "B" + slen).Value2 <- namesVert 
    worksheet.Range("C3", "E" + slen).Value2 <- tableArr

    // Add new item to the charts collection
    let chartobjects = (worksheet.ChartObjects() :?> ChartObjects) 
    let chartobject = chartobjects.Add(400.0, 20.0, 550.0, 350.0) 

    // Configure the chart using the wizard
    chartobject.Chart.ChartWizard
      (Title = "Sample Data",
       Source = worksheet.Range("B2", "E" + slen),
       Gallery = XlChartType.xl3DColumn, PlotBy = XlRowCol.xlColumns,
       SeriesLabels = 1, CategoryLabels = 1,
       CategoryTitle = "", ValueTitle = "Some value")

    // Set graphical style of the chart
    chartobject.Chart.ChartStyle <- 21

