from pdf2image import convert_from_path
import os
import pytesseract
import cv2
import time
import fitz

"""With pytesseract library we also need to install tesseract-ocr for windows. 
More info: https://github.com/UB-Mannheim/tesseract/wiki.
Then, we need to download language data set from: https://github.com/tesseract-ocr/langdata
and test data set from: https://github.com/tesseract-ocr/tessdata.
Unzip the file and copy-paste the folders to tesseract installation path;
e.x. C:\Program Files\Tesseract-OCR
test data and language data folders must have the names "langdata" and "tessdata".
Finally, configure tessdata path as shown below and use it as a parameter when tesseract is called in the code."""


# Set the tesseract test data path in the code
tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
# Set the tesseract path in the code
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# Poppler dependecy is needed for pdf2image to work
poppler = r'C:\Program Files (x86)\poppler-0.68.0\bin'
# Extract images from pdf and save to a directory
pdf_path = r'C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/pdfs/Κυλινδρόμυλοι Α.Ε. κλπ. απόφαση 954 ΚΠολΔ.pdf'
directory = 'C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/pages/'


def pdf_to_txt_pymupdf(pdf_path):
    start = time.time()

    # open your file
    doc = fitz.open(pdf_path)
    # iterate through the pages of the document and create a RGB image of the page
    for page in doc:
        pix = page.get_pixmap()
        pix.save("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/pages/page-%i.png" % page.number)


    # Save full paths of pages in a list
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    text_name_from_pdf = 'C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/texts/' + \
                         os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
    # A text file is created and flushed
    txt_file = open(text_name_from_pdf, "w", encoding="utf-8")
    # OpenCV will handle image processing
    for path in file_paths:
        img = cv2.imread(path)
        # Rescaling the image (it's recommended if you’re working with images that have a DPI of less than 300 dpi)
        img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        # Convert the image to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Applying dilation and erosion to remove the noise (you may play with the kernel size depending on your data set)
        #kernel = np.ones((1, 1), np.uint8)
        #img = cv2.dilate(img, kernel, iterations=1)
        #img = cv2.erode(img, kernel, iterations=1)
        # Applying blur, which can be done by using one of the following lines
        # (each of which has its pros and cons, however, median blur and bilateral filter usually perform better than gaussian blur.)
        #img = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        #img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        #img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        if img is None:
            txt_file = open(text_name_from_pdf, "a", encoding="utf-8")
            text = "---Could not read the image.---"
            txt_file.write(text)
            txt_file.close()
        else:
            txt_file = open(text_name_from_pdf, "a", encoding="utf-8")
            text = pytesseract.image_to_string(img, lang="ell", config=tessdata_dir_config)
            txt_file.write(text)
            txt_file.close()
    end = time.time()
    print('Pdf was converted to txt successfully.')
    print("Pdf to text conversion took {:.1f} seconds".format(end - start))
    print(text_name_from_pdf)
    return text_name_from_pdf


def pdf_to_txt_pdf2image(pdf_path):
    start = time.time()
    pages = convert_from_path(pdf_path, 500, poppler_path=poppler)
    count = 0
    for page in pages:
        page.save('C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/pages/' + 'page-%d.png' % count, 'PNG')
        count += 1

    # Save full paths of pages in a list
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    text_name_from_pdf = 'C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/texts/' + \
                         os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
    # A text file is created and flushed
    txt_file = open(text_name_from_pdf, "w", encoding="utf-8")
    # OpenCV will handle image processing
    for path in file_paths:
        img = cv2.imread(path)
        # Rescaling the image (it's recommended if you’re working with images that have a DPI of less than 300 dpi)
        img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        # Convert the image to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Applying dilation and erosion to remove the noise (you may play with the kernel size depending on your data set)
        #kernel = np.ones((1, 1), np.uint8)
        #img = cv2.dilate(img, kernel, iterations=1)
        #img = cv2.erode(img, kernel, iterations=1)
        # Applying blur, which can be done by using one of the following lines
        # (each of which has its pros and cons, however, median blur and bilateral filter usually perform better than gaussian blur.)
        #img = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        #img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        #img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        if img is None:
            txt_file = open(text_name_from_pdf, "a", encoding="utf-8")
            text = "---Could not read the image.---"
            txt_file.write(text)
            txt_file.close()
        else:
            txt_file = open(text_name_from_pdf, "a", encoding="utf-8")
            text = pytesseract.image_to_string(img, lang="ell", config=tessdata_dir_config)
            txt_file.write(text)
            txt_file.close()
    end = time.time()
    print('Pdf was converted to txt successfully.')
    print("Pdf to text conversion took {:.1f} seconds".format(end - start))
    print(text_name_from_pdf)
    return text_name_from_pdf

pdf_to_txt_pymupdf(pdf_path)