$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$reports = Join-Path $root "reports"
$documents = Get-ChildItem -LiteralPath $reports -Filter "*Report_FA.docx" -File | Sort-Object Name
if (-not $documents) {
    throw "No report DOCX files found in $reports"
}

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($source in $documents) {
        $document = $word.Documents.Open($source.FullName, $false, $false)
        try {
            foreach ($toc in $document.TablesOfContents) {
                $toc.Update() | Out-Null
            }
            $document.Fields.Update() | Out-Null
            foreach ($style in $document.Styles) {
                $style.Font.Name = "B Nazanin"
                $style.Font.NameAscii = "B Nazanin"
                $style.Font.NameBi = "B Nazanin"
                $style.Font.NameOther = "B Nazanin"
            }
            foreach ($storyType in $document.StoryRanges) {
                $story = $storyType
                while ($null -ne $story) {
                    $story.Font.Name = "B Nazanin"
                    $story.Font.NameAscii = "B Nazanin"
                    $story.Font.NameBi = "B Nazanin"
                    $story.Font.NameOther = "B Nazanin"
                    $story = $story.NextStoryRange
                }
            }
            foreach ($paragraph in $document.Paragraphs) {
                if ($paragraph.Range.Text -match '[\u0600-\u06FF]') {
                    $paragraph.Range.ParagraphFormat.ReadingOrder = 0
                }
            }
            foreach ($table in $document.Tables) {
                $table.TableDirection = 0
            }
            $document.Save()
            $target = [System.IO.Path]::ChangeExtension($source.FullName, ".pdf")
            $document.ExportAsFixedFormat($target, 17)
            Write-Output "Exported $([System.IO.Path]::GetFileName($target))"
        }
        finally {
            $document.Close(0)
        }
    }
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
