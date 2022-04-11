import os
import pytesseract
import cv2
import time
import fitz
import datetime
import shutil
import pathlib

"""With pytesseract library we also need to install tesseract-ocr for windows. 
More info: https://github.com/UB-Mannheim/tesseract/wiki.
Then, we need to download language data set from: https://github.com/tesseract-ocr/langdata
and test data set from: https://github.com/tesseract-ocr/tessdata.
Unzip the file and copy-paste the folders to tesseract installation path;
e.x. C:\Program Files\Tesseract-OCR
test data and language data folders must have the names "langdata" and "tessdata".
Finally, configure tessdata path as shown below and use it as a parameter when tesseract is called in the code."""

start = time.time()
# Set the tesseract test data path in the code
tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
# Set the tesseract path in the code
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# Poppler dependecy is needed for pdf2image to work
poppler = r'C:\Program Files (x86)\poppler-0.68.0\bin'
# Extract images from pdf and save to a directory
directory = 'C:\\Users\\kostas.skepetaris\\PycharmProjects\\object_detection_nlp\\app\\static\\pages'
pages = 'C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/pages'


def pdf_to_txt_pymupdf(pdf_filepath):
    start = time.time()
    # Save full paths of pages in a list
    file_paths = []  # List which will store all of the full filepaths.

    # open your file
    doc = fitz.open(pdf_filepath)
    # iterate through the pages of the document and create a RGB image of the page
    for page in doc:
        zoom = 3  # zoom factor
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        pix.save("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/pages/page-%i.png" % page.number)

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    text_name_from_pdf = 'C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/texts/' + \
                         os.path.splitext(os.path.basename(pdf_filepath))[0] + '.txt'
    # A text file is created and flushed
    txt_file = open(text_name_from_pdf, "w", encoding="utf-8")
    # OpenCV will handle image processing
    for path in file_paths:
        img = cv2.imread(path)
        # Rescaling the image (it's recommended if you’re working with images that have a DPI of less than 300 dpi)
        img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        # Convert the image to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Applying dilation and erosion to remove the noise (you may play with the kernel size depending on your data set)
        # kernel = np.ones((1, 1), np.uint8)
        # img = cv2.dilate(img, kernel, iterations=1)
        # img = cv2.erode(img, kernel, iterations=1)
        # Applying blur, which can be done by using one of the following lines
        # (each of which has its pros and cons, however, median blur and bilateral filter usually perform better than gaussian blur.)
        # img = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        # img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        # img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        if img is None:
            txt_file = open(text_name_from_pdf, "a", encoding="utf-8")
            text = "---Could not read the image.---"
            txt_file.write(text)
            txt_file.close()
        else:
            txt_file = open(text_name_from_pdf, "a", encoding="utf-8")
            text = pytesseract.image_to_string(img, lang="ell+eng", config=tessdata_dir_config)
            txt_file.write(text)
            txt_file.close()

    for root, dirs, files in os.walk(pages):
        try:
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        except OSError as e:
            print("Error: %s : %s" % (files, e.strerror))
    end = time.time()
    created_in = datetime.datetime.fromtimestamp(pathlib.Path(pdf_filepath).stat().st_ctime, tz=datetime.timezone.utc)
    print("Pdf was converted to txt successfully in {:.1f} seconds".format(end - start))
    print("Txt created in:", created_in)
    print(os.path.getmtime(pdf_filepath))
    return text_name_from_pdf


'''for root, directories, files in os.walk('C:/Users/kostas.skepetaris/PycharmProjects/web_retrieve_data/static/pdfs/deltio_nomikon'):
    for filename in files:
        # Join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        if os.path.getmtime(filepath) >= 1649053866.0058098:
            pdf_to_txt_pymupdf(filepath)'''


pdf_filepath = 'C:/Users/kostas.skepetaris/PycharmProjects/web_retrieve_data/static/pdfs/deltio_nomikon/report179954.pdf'
pdf_to_txt_pymupdf(pdf_filepath)
print('Job finished!')