from PyPDF2 import PdfFileMerger, PdfFileReader
import os
def merge_pdf(directory_path):

    # Call the PdfFileMerger
    mergedObject = PdfFileMerger()
    
    for filename in os.listdir(directory_path):
        mergedObject.append(PdfFileReader(os.path.join(directory_path, filename), 'rb'))

    mergedObject.write("../outputs/mergedfilesoutput.pdf")