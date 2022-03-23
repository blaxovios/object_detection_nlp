"""Custom Named Entity Recognition (NER) and feature extraction with NLP
This module reads xlsx file that contains all text from auction files downloaded as pdf from deltio nomikon. We will manually
make it detect entities based on our own category of words, leveraging NER to identify them. Annotating the data,
preparation of the training data, training a custom spaCy v3 model (previous versions of spacy differ a lot) and assess
model results are all the necessary steps.
Spacy will be used for those processes.

Author: Kostas

Created: 22nd March, 2022
"""
# Imports
import pandas as pd
import spacy
from spacy import displacy
from spacy.tokens import DocBin
import json
from datetime import datetime
from tqdm import tqdm
import re
from os import listdir
from os.path import isfile, join
import os
from IPython.core.display import display, HTML


# With the help of: https://levelup.gitconnected.com/auto-detect-anything-with-custom-named-entity-recognition-ner-c89d6562e8e9

'''text_filepath = 'C:/Users/kostas.skepetaris/PycharmProjects/web_retrieve_data/static/results/deltio_nomikon/deltio 08-03-2022.xlsx'
text_df = pd.read_excel(text_filepath)
labels_df = pd.read_excel('C:/Users/kostas.skepetaris/PycharmProjects/web_retrieve_data/static/csv/Weekly update 11012022_17022022.xlsx')
labels_list = list(labels_df.columns)
text_df = text_df.reindex(columns=[*text_df.columns.tolist(), 'labels'], fill_value=str(labels_list))
text_description_labels_df = text_df[['Asset Description', 'labels']]
text_description_labels_df = text_description_labels_df[~text_description_labels_df['Asset Description'].duplicated()]
asset_description_df = text_description_labels_df['Asset Description']'''

mypath = "C:\\Users\\kostas.skepetaris\\PycharmProjects\\web_retrieve_data\\static\\texts"
txt_filepaths = []
def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            txt_filepaths.append(os.path.abspath(os.path.join(dirpath, f)))
	return txt_filepaths

absoluteFilePaths(mypath)

# this dictionary will contain all annotated examples
collective_dict = {'TRAINING_DATA': []}


def structure_training_data(text, kw_auction_date_list, kw_tax_id_of_debtor_list, kw_auction_id_list, kw_eauction_website_list,
							kw_foreclosure_id_list, kw_accelerator_list, kw_tax_id_accelerator_list, kw_property_type_list,
							kw_size_of_land_list, kw_percentage_of_ownership_list, kw_notary_public_list):
	results = []
	entities = []

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_auction_date_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "auction_date"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_tax_id_of_debtor_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "tax_id_of_debtor"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_auction_id_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "auction_id"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_eauction_website_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "eauction_website"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_foreclosure_id_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "foreclosure_id"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_accelerator_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "accelerator"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_tax_id_accelerator_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "tax_id_accelerator"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_property_type_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "property_type"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_size_of_land_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "size_of_land"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_percentage_of_ownership_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "percentage_of_ownership"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_notary_public_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "notary_public"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# add any found entities into a JSON format within collective_dict
	if len(entities) > 0:
		results = [text, {"entities": entities}]
		collective_dict['TRAINING_DATA'].append(results)
		return


# Define training keywords and input text
kw_auction_date_list = ['Ημερομηνία Διεξαγωγής Πλειστηριασμού: 23/03/2022']
kw_tax_id_of_debtor_list = ['ΑΦΜ Οφειλέτη: 036044215']
kw_auction_id_list = ['Μοναδικός Κωδικός']
kw_eauction_website_list = ['ΙΣΤΟΣΕΛΙΔΑ ΔΗΜΟΣΙΕΥΣΕΩΝ ΠΛΕΙΣΤΗΡΙΑΣΜΩΝ']
kw_foreclosure_id_list = ['έκθεση αναγκαστικής κατάσχεσης']
kw_accelerator_list = ['Επισπεύδων']
kw_tax_id_accelerator_list = ['ΑΦΜ Επισπεύδοντα']
kw_property_type_list = ['διαμέρισμα', 'αποθήκη', 'μεζονέτα', 'μονοκατοικία', 'διώροφη οικοδομή διαμερισμάτων',
						 'διώροφη οικοδομή', 'διόροφη οικοδομή', 'διαμερίσματα', 'καταστημάτων', 'καταστήματα',
						 'Επαγγελματικό Κτίριο', 'Κτήριο', 'Ξενοδοχειακή Μονάδα', 'Αποθηκευτικός Χώρος', 'Βιομηχανικός Χώρος',
						 'Βιοτεχνικός Χώρος', 'Αίθουσα', 'Εκθεσιακός Χώρος', 'Υπόγειο parking', 'Κλειστό parking',
						 'Parking πιλοτής', 'χώροι σταύθμευσης', 'χώρος στάθμευσης', 'Ανοιχτό Parking', 'ξενοδοχείο',
						 'Κατοικία-διαμέρισμα', 'Επαγγελματική στέγη', 'Γεωργικά κτίρια', 'Γεωργικό κτίριο', 'Γεωργικά κτήρια',
						 'Γεωργικό κτήρια', 'κτηνοτροφικά κτίρια', 'κτηνοτροφικά κτήρια', 'κτηνοτροφικό κτίριο',
						 'κτηνοτροφικό κτήριο', 'Θέσης στάθμευσης', 'Θέσεις στάθμευσης', 'Θέση στάθμευσης',
						 'Θέσεων στάθμευσης', 'Βιομηχανικά κτήρια', 'Βιομηχανικά κτίρια', 'Βιομηχανικοί χώροι',
						 'Βιομηχανικών Χώρων', 'τουριστικές εγκαταστάσεις', 'τουριστικών εγκαταστάσεων', 'Εκπαιδευτήρια',
						 'Εκπαιδευτήριο', 'Αθλητικές εγκαταστάσεις', 'Αθλητικών εγκαταστάσεων']
kw_size_of_land_list = ['οικοπέδου', 'τμήμα οικοπέδου εμβαδού', 'τμήμα οικοπέδου επιφάνειας',
						'αποτελεί τμήμα μεγαλύτερου οικοπέδου ολικού εμβαδού', 'αποτελεί τμήμα μεγαλύτερου οικοπέδου',
						'επί οικοπέδου εκτάσεως', 'κείται επί οικοπέδου εκτάσεως', 'μετά του οικοπέδου εκτάσεως',
						'επί οικοπέδου εκτάσεως', 'επί του οικοπέδου εκτάσεως', 'το κτίσμα είναι σε οικόπεδο',
						'ανήκει σε οικόπεδο', 'κυριότητας ενός οικοπέδου εκτάσεως', 'συνιδιοκτησίας επί του οικοπέδου',
						'ιδιοκτησίας επί του οικοπέδου', 'κτισμένη πάνω σε οικόπεδο μ.τ.', 'κτισμένη πάνω σε οικόπεδο']
kw_percentage_of_ownership_list = ['δικαιώματος της πλήρους κυριότητας σε ποσοστό', 'πλήρης κυριότητας σε ποσοστό',
								   'πλήρη κυριότητα σε ποσοστό', 'ποσοστό συνιδιοκτησίας εξ αδιαιρέτου',
								   'συμμετέχει στο κοινό οικόπεδο με ποσοστό συνιδιοκτησίας']
kw_notary_public_list = ['Η Συμβολαιογράφος', 'O Συμβολαιογράφος']

# Train each text seperately
for i in txt_filepaths[:30]:
	with open(i, 'r',
			  encoding="utf8") as f:
		data = f.read().replace('\n', ' ')
	structure_training_data(data, kw_auction_date_list, kw_tax_id_of_debtor_list, kw_auction_id_list, kw_eauction_website_list,
							kw_foreclosure_id_list, kw_accelerator_list, kw_tax_id_accelerator_list, kw_property_type_list,
							kw_size_of_land_list, kw_percentage_of_ownership_list, kw_notary_public_list)

collective_dict

# define our training data to TRAIN_DATA
TRAIN_DATA = collective_dict['TRAINING_DATA']

# create a blank model
nlp = spacy.blank('en')

def create_training(TRAIN_DATA):
    db = DocBin()
    for text, annot in tqdm(TRAIN_DATA):
        doc = nlp.make_doc(text)
        ents = []

        # create span objects
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")

            # skip if the character indices do not map to a valid span
            if span is None:
                print("Skipping entity.")
            else:
                ents.append(span)
                # handle erroneous entity annotations by removing them
                try:
                    doc.ents = ents
                except:
                    # print("BAD SPAN:", span, "\n")
                    ents.pop()
        doc.ents = ents

        # pack Doc objects into DocBin
        db.add(doc)
    return db

TRAIN_DATA_DOC = create_training(TRAIN_DATA)

# Export results (here I add it to a TRAIN_DATA folder within the directory)
TRAIN_DATA_DOC.to_disk("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/train_data/train_data.spacy")

"""
1. Follow the link: https://spacy.io/usage/training#config and selecte ‘ner’ in the Components section of the widget. 
Copy/paste the full contents of the widget into a file named base_config.cfg within your folder directory.

2. Open the base_config.cfg file. All you need to change is the train and dev variables at the top.
Dev is reserved for the validation dataset. However, because I just want to train the model for demonstration purposes, 
I added the same path to both train and dev. (Note: this will result in 100% training accuracy)

3. Open your CLI and cd over into the directory of base_config.cfg and simply copy/paste this command:
python -m spacy init fill-config base_config.cfg config.cfg

4. Run the following to begin training: 
python -m spacy train config.cfg --output C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/results
"""

# Test out the model we trained on a brand new piece of text.
with open(txt_filepaths[67], 'r', encoding="utf8") as f:
	test_1 = f.read().replace('\n', ' ')

# load the trained model
nlp_output = spacy.load("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/results/model-best")

# pass our test instance into the trained pipeline
doc = nlp_output(test_1)

'''# customize the label colors
colors = {"auction_date": "linear-gradient(90deg, #E1D436, #F59710)", "tax_id_of_debtor": "linear-gradient(90deg, #E1D436, #F59710)"}
options = {"ents": ["auction_date", "tax_id_of_debtor"], "colors": colors}

# visualize the identified entities
html = displacy.render(doc, style="ent", options=options)'''

# print out the identified entities
for ent in doc.ents:
    if ent.label_ == "auction_date":
        print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "tax_id_of_debtor":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "auction_id":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "eauction_website":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "foreclosure_id":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "accelerator":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "tax_id_accelerator":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "property_type":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "size_of_land":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "percentage_of_ownership":
		print(ent.text, ent.label_)
for ent in doc.ents:
	if ent.label_ == "notary_public":
		print(ent.text, ent.label_)