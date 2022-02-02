from pdf2image import convert_from_path
import os
import pytesseract
import cv2
import time
import pandas as pd

'''With pytesseract library we also need to install tesseract-ocr for windows. 
More info: https://github.com/UB-Mannheim/tesseract/wiki.
Then, we need to download language data set from: https://github.com/tesseract-ocr/langdata
and test data set from: https://github.com/tesseract-ocr/tessdata.
Unzip the file and copy-paste the folders to tesseract installation path;
e.x. C:\Program Files\Tesseract-OCR
test data and language data folders must have the names "langdata" and "tessdata".
Finally, configure tessdata path as shown below and use it as a parameter when tesseract is called in the code.'''

# Set the tesseract test data path in the code
tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
# Set the tesseract path in the code
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Poppler dependecy is needed for pdf2image to work
poppler = r'C:\Program Files (x86)\poppler-0.68.0\bin'
# Extract images from pdf and save to a directory
pdf_path = 'C:/Users/tsepe/PycharmProjects/objectDetectionNlp/app/static/pdfs/Κυλινδρόμυλοι Α.Ε. κλπ. απόφαση 954 ΚΠολΔ.pdf'
start1 = time.time()


def convert_pdf(pdf_path, poppler):
    pages = convert_from_path(pdf_path, 500, poppler_path=poppler)
    count = 0
    for page in pages:
        page.save(
            'C:/Users/tsepe/PycharmProjects/objectDetectionNlp/app/static/pages/' + 'page%d.jpg' % count,
            'JPEG')
        count += 1


convert_pdf(pdf_path, poppler)
end1 = time.time()
print("Pdf to image conversion took {:.1f} seconds".format(end1 - start1))
# Save full paths of pages in a list
directory = 'C:/Users/tsepe/PycharmProjects/objectDetectionNlp/app/static/pages/'


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

text_name_from_pdf = 'C:/Users/tsepe/PycharmProjects/objectDetectionNlp/app/static/texts/' + \
                     os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
start2 = time.time()


def image_to_text(text_name_from_pdf):
    # A text file is created and flushed
    txt_file = open(text_name_from_pdf, "w", encoding="utf-8")
    # OpenCV will handle image processing
    for path in full_file_paths:
        img = cv2.imread(path)
        # Convert the image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            txt_file = open(text_name_from_pdf, "a", encoding="utf-8")
            text = "---Could not read the image.---"
            txt_file.write(text)
            txt_file.close()
        else:
            txt_file = open(text_name_from_pdf, "a", encoding="utf-8")
            text = pytesseract.image_to_string(gray, lang="ell", config=tessdata_dir_config)
            txt_file.write(text)
            txt_file.close()
    return txt_file


image_to_text(text_name_from_pdf)
end2 = time.time()
print("Image to text conversion took {:.1f} seconds".format(end2 - start2))
print('Pdf was extracted to text in'.format(end2 - start1))
