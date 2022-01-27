from pdf2image import convert_from_path
import os
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Poppler dependecy is needed for pdf2image to work
poppler = r'C:\Program Files (x86)\poppler-0.68.0\bin'
pages = convert_from_path(r'C:\Users\tsepe\PycharmProjects\objectDetectionNlp\app\static\pdfs\Κυλινδρόμυλοι Α.Ε. κλπ. απόφαση 954 ΚΠολΔ.pdf',
                          500, poppler_path=poppler)

# Extracte images from pdf and save to a directory
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

for path in full_file_paths:
    img = cv2.imread(path)

    img = cv2.resize(img, (600, 360))
    print(pytesseract.image_to_string(img))