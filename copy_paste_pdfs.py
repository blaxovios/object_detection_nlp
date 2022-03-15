import os, shutil
import win32com.client


def copy_paste_pdfs():
    destination = r'C:\Users\kostas.skepetaris\Desktop\KOSTAS\2_NLP\1_Greek Office\1_Terra Project\Επιθεωρήσεις_All_Pdfs'
    source = r'C:\Users\kostas.skepetaris\Desktop\KOSTAS\2_NLP\1_Greek Office\1_Terra Project\Επιθεωρήσεις_As_Received'

    # This will walk a tree with sub-directories. You can do an os.path.isfile check to make it a little safer.
    for root, dirs, files in os.walk(source):
        for file in files:
            if os.path.normcase(file[-4:]) == '.pdf':
                shutil.copy(os.path.join(root, file), os.path.join(destination, file))     # Joins path with filename
    print('Done copying pasting pdfs.')


def copy_paste_excel():
    destination_excel = r'C:\Users\kostas.skepetaris\Desktop\KOSTAS\2_NLP\1_Greek Office\1_Terra Project\Επιθεωρήσεις_All_Excel'
    source = r'C:\Users\kostas.skepetaris\Desktop\KOSTAS\2_NLP\1_Greek Office\1_Terra Project\Επιθεωρήσεις_As_Received'

    # This will walk a tree with sub-directories. You can do an os.path.isfile check to make it a little safer.
    for root, dirs, files in os.walk(source):
        for file in files:
            if (file.endswith('.xlsx') or file.endswith('.xls')):
                shutil.copy(os.path.join(root, file), os.path.join(destination_excel, file))     # Joins path with filename without extension

    return print('Done copying pasting excels.')


def excel_to_pdfs():
    o = win32com.client.Dispatch("Excel.Application")
    o.Visible = False
    o.SendKeys("{Enter}", Wait=1)
    o.DisplayAlerts = 0
    o.Interactive = False
    o.Application.EnableEvents = False
    o.DisplayAlerts = False  # Suppress any Alert windows, which require User action
    o.AskToUpdateLinks = False  # Disable automatic update linking
    destination = r'C:\Users\kostas.skepetaris\Desktop\KOSTAS\2_NLP\1_Greek Office\1_Terra Project\Επιθεωρήσεις_All_Excel_to_Pdfs'
    source_excel = r'C:\Users\kostas.skepetaris\Desktop\KOSTAS\2_NLP\1_Greek Office\1_Terra Project\Επιθεωρήσεις_All_Excel'
    for root, dirs, files in os.walk(source_excel):
        for file in files:
            if (file.endswith('.xlsx') or file.endswith('.xls')):
                excel_file = os.path.join(root, file)
                try:
                    # Load Excel file
                    wb = o.Workbooks.Open(excel_file)
                    # Convert Excel to PDF
                    wb.ActiveSheet.ExportAsFixedFormat(0, (os.path.join(destination, os.path.splitext(file)[0]+'.pdf')))
                    wb.Close(True)
                except:
                    continue
    return print('Done converting excels to pdfs and copying pasting them.')

excel_to_pdfs()