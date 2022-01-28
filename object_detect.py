from pdf2image import convert_from_path
import os
import pytesseract
import cv2
import pandas as pd

'''With pytesseract library we also need to install tesseract-ocr for windows. 
More info: https://github.com/UB-Mannheim/tesseract/wiki.
Then, we need to download language data set from: https://github.com/tesseract-ocr/langdata
and test data set from: https://github.com/tesseract-ocr/tessdata.
Unzip the file and copy-paste the folders to tesseract installation path;
e.x. C:\Program Files\Tesseract-OCR
test data and language data folders must have the names "langdata" and "tessdata".
Finally, configure tessdata path as shown below and use it as a parameter when tesseract is called in the code.'''

tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
# Set the tesseract path in the code
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Poppler dependecy is needed for pdf2image to work
poppler = r'C:\Program Files (x86)\poppler-0.68.0\bin'

# Extract images from pdf and save to a directory
pages = convert_from_path(r'C:\Users\tsepe\PycharmProjects\objectDetectionNlp\app\static\pdfs\Κυλινδρόμυλοι Α.Ε. κλπ. απόφαση 954 ΚΠολΔ.pdf',
                          500, poppler_path=poppler)
count = 0
for page in pages:
    page.save('C:/Users/tsepe/PycharmProjects/objectDetectionNlp/app/static/pages/'+'page%d.jpg' % count, 'JPEG')
    count += 1
print('Pages were extracted from pdf')

# Save full paths of pages in a list
directory = ('C:/Users/tsepe/PycharmProjects/objectDetectionNlp/app/static/pages/')
def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths
full_file_paths = get_filepaths(directory)
print(full_file_paths)

pages_to_text_list = []
for path in full_file_paths:
    img = cv2.imread(path)
    if img is None:
        pages_to_text_list.append("Could not read the image.")

    else:
        rev = pytesseract.image_to_string(img, lang="ell", config=tessdata_dir_config)
        pages_to_text_list.append(rev)
print(pages_to_text_list)
df_pages_to_text = pd.DataFrame(pages_to_text_list, columns=['Description'])
df_pages_to_text.to_excel(r'C:\Users\tsepe\PycharmProjects\objectDetectionNlp\app\static\results\pages_to_text.xlsx', encoding='utf-8', index=False)