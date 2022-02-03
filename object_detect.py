import numpy as np
from pdf2image import convert_from_path
import os
import pytesseract
import cv2
import time
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split

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
pdf_path = 'C:/Users/kostas.skepetaris/PycharmProjects/objectDetectionNlp/app/static/pdfs/Κυλινδρόμυλοι Α.Ε. κλπ. απόφαση 954 ΚΠολΔ.pdf'
start1 = time.time()


def convert_pdf(pdf_path, poppler):
    pages = convert_from_path(pdf_path, 500, poppler_path=poppler)
    count = 0
    for page in pages:
        page.save(
            'C:/Users/kostas.skepetaris/PycharmProjects/objectDetectionNlp/app/static/pages/' + 'page%d.jpg' % count,
            'JPEG')
        count += 1


convert_pdf(pdf_path, poppler)
end1 = time.time()
print("Pdf to image conversion took {:.1f} seconds".format(end1 - start1))
# Save full paths of pages in a list
directory = 'C:/Users/kostas.skepetaris/PycharmProjects/objectDetectionNlp/app/static/pages/'


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

text_name_from_pdf = 'C:/Users/kostas.skepetaris/PycharmProjects/objectDetectionNlp/app/static/texts/' + \
                     os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
start2 = time.time()


def image_to_text(text_name_from_pdf):
    # A text file is created and flushed
    txt_file = open(text_name_from_pdf, "w", encoding="utf-8")
    # OpenCV will handle image processing
    for path in full_file_paths:
        img = cv2.imread(path)
        # Rescaling the image (it's recommended if you’re working with images that have a DPI of less than 300 dpi)
        img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        # Convert the image to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Applying dilation and erosion to remove the noise (you may play with the kernel size depending on your data set)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        # Applying blur, which can be done by using one of the following lines
        # (each of which has its pros and cons, however, median blur and bilateral filter usually perform better than gaussian blur.)
        #final_image = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #final_image = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #final_image = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #final_image = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        #final_image = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        #final_image = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
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
    return txt_file


image_to_text(text_name_from_pdf)
end2 = time.time()
print("Image to text conversion took {:.1f} seconds".format(end2 - start2))
print('Pdf was extracted to text in'.format(end2 - start1))

print('Loading data')
file = open(text_name_from_pdf).read()

x, y, vocabulary, vocabulary_inv = file

# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765

X_train, X_test, y_train, y_test = train_test_split( x, y,     test_size=0.2, random_state=42)

# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)


sequence_length = x.shape[1] # 56
vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

epochs = 100
batch_size = 30

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size,     output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0],     embedding_dim), padding='valid', kernel_initializer='normal',     activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1],     embedding_dim), padding='valid', kernel_initializer='normal',     activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2],     embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5',     monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  #     starts training