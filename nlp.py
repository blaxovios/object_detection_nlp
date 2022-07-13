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
import numpy as np
import spacy
from spacy.tokens import DocBin
import json
from datetime import datetime
from tqdm import tqdm
import re
from os import listdir
from os.path import isfile, join
import os
import random


#
# class AutoAnnotateBasedOnSimilarKeyword:
#     # With the help of: https://levelup.gitconnected.com/auto-detect-anything-with-custom-named-entity-recognition-ner-c89d6562e8e9
#
#
#     def absoluteFilePaths(self):
#         # Text files from auction pdfs
#         mypath = "C:\\Users\\kostas.skepetaris\\PycharmProjects\\web_retrieve_data\\static\\texts"
#         txt_filepaths = []
#         for dirpath, _, filenames in os.walk(mypath):
#             for f in filenames:
#                 txt_filepaths.append(os.path.abspath(os.path.join(dirpath, f)))
#         return txt_filepaths
#
#     '''First, we need to convert our training examples from JSON format into spaCy Doc objects.
#     The Doc objects are then stored in a collective DocBin class. This is just the new format that spaCy requires for training.'''
#
#
#     def keywords(self, txt_filepaths):
#         # This dictionary will contain all annotated examples to be used for training data
#         collective_dict = {'TRAINING_DATA': []}
#
#         # Define training keywords and input text
#         kw_auction_date_list = ['Ημερομηνία Διεξαγωγής Πλειστηριασμού']
#         kw_tax_id_of_debtor_list = ['ΑΦΜ Οφειλέτη']
#         kw_auction_id_list = ['Μοναδικός Κωδικός']
#         kw_eauction_website_list = ['ΙΣΤΟΣΕΛΙΔΑ ΔΗΜΟΣΙΕΥΣΕΩΝ ΠΛΕΙΣΤΗΡΙΑΣΜΩΝ']
#         kw_foreclosure_id_list = ['έκθεση αναγκαστικής κατάσχεσης']
#         kw_accelerator_list = ['Επισπεύδων']
#         kw_tax_id_accelerator_list = ['ΑΦΜ Επισπεύδοντα']
#         kw_property_type_list = ['διαμέρισμα', 'αποθήκη', 'μεζονέτα', 'μονοκατοικία', 'διώροφη οικοδομή διαμερισμάτων',
#                                  'διώροφη οικοδομή', 'διόροφη οικοδομή', 'διαμερίσματα', 'καταστημάτων', 'καταστήματα',
#                                  'Επαγγελματικό Κτίριο', 'Κτήριο', 'Ξενοδοχειακή Μονάδα', 'Αποθηκευτικός Χώρος',
#                                  'Βιομηχανικός Χώρος',
#                                  'Βιοτεχνικός Χώρος', 'Αίθουσα', 'Εκθεσιακός Χώρος', 'Υπόγειο parking',
#                                  'Κλειστό parking',
#                                  'Parking πιλοτής', 'χώροι σταύθμευσης', 'χώρος στάθμευσης', 'Ανοιχτό Parking',
#                                  'ξενοδοχείο',
#                                  'Κατοικία-διαμέρισμα', 'Επαγγελματική στέγη', 'Γεωργικά κτίρια', 'Γεωργικό κτίριο',
#                                  'Γεωργικά κτήρια',
#                                  'Γεωργικό κτήρια', 'κτηνοτροφικά κτίρια', 'κτηνοτροφικά κτήρια', 'κτηνοτροφικό κτίριο',
#                                  'κτηνοτροφικό κτήριο', 'Θέσης στάθμευσης', 'Θέσεις στάθμευσης', 'Θέση στάθμευσης',
#                                  'Θέσεων στάθμευσης', 'Βιομηχανικά κτήρια', 'Βιομηχανικά κτίρια', 'Βιομηχανικοί χώροι',
#                                  'Βιομηχανικών Χώρων', 'τουριστικές εγκαταστάσεις', 'τουριστικών εγκαταστάσεων',
#                                  'Εκπαιδευτήρια',
#                                  'Εκπαιδευτήριο', 'Αθλητικές εγκαταστάσεις', 'Αθλητικών εγκαταστάσεων']
#         kw_size_of_land_list = ['οικοπέδου', 'τμήμα οικοπέδου εμβαδού', 'τμήμα οικοπέδου επιφάνειας',
#                                 'αποτελεί τμήμα μεγαλύτερου οικοπέδου ολικού εμβαδού',
#                                 'αποτελεί τμήμα μεγαλύτερου οικοπέδου',
#                                 'επί οικοπέδου εκτάσεως', 'κείται επί οικοπέδου εκτάσεως',
#                                 'μετά του οικοπέδου εκτάσεως',
#                                 'επί οικοπέδου εκτάσεως', 'επί του οικοπέδου εκτάσεως', 'το κτίσμα είναι σε οικόπεδο',
#                                 'ανήκει σε οικόπεδο', 'κυριότητας ενός οικοπέδου εκτάσεως',
#                                 'συνιδιοκτησίας επί του οικοπέδου',
#                                 'ιδιοκτησίας επί του οικοπέδου', 'κτισμένη πάνω σε οικόπεδο μ.τ.',
#                                 'κτισμένη πάνω σε οικόπεδο']
#         kw_percentage_of_ownership_list = ['δικαιώματος της πλήρους κυριότητας σε ποσοστό',
#                                            'πλήρης κυριότητας σε ποσοστό',
#                                            'πλήρη κυριότητα σε ποσοστό', 'ποσοστό συνιδιοκτησίας εξ αδιαιρέτου',
#                                            'συμμετέχει στο κοινό οικόπεδο με ποσοστό συνιδιοκτησίας']
#         kw_notary_public_list = ['Η Συμβολαιογράφος', 'O Συμβολαιογράφος']
#
#         # Loop txt files to use as train data
#         for i in txt_filepaths[:100]:
#             with open(i, 'r', encoding="utf8") as f:
#                 f.read().replace('\n', ' ')
#
#                 # define our training data to TRAIN_DATA
#         TRAIN_DATA = collective_dict['TRAINING_DATA']
#
#         # create a blank model
#         nlp = spacy.blank("xx")
#         return kw_auction_date_list, kw_tax_id_of_debtor_list, kw_auction_id_list, kw_eauction_website_list, \
#                kw_foreclosure_id_list, kw_accelerator_list, kw_tax_id_accelerator_list, kw_property_type_list, \
#                kw_size_of_land_list, kw_percentage_of_ownership_list, kw_notary_public_list, TRAIN_DATA, nlp
#
#
#     def structure_training_data(text, kw_auction_date_list, kw_tax_id_of_debtor_list, kw_auction_id_list, kw_eauction_website_list,
#                                 kw_foreclosure_id_list, kw_accelerator_list, kw_tax_id_accelerator_list, kw_property_type_list,
#                                 kw_size_of_land_list, kw_percentage_of_ownership_list, kw_notary_public_list, TRAIN_DATA):
#
#         entities = []
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_auction_date_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "auction_date"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_tax_id_of_debtor_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "tax_id_of_debtor"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_auction_id_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "auction_id"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_eauction_website_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "eauction_website"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_foreclosure_id_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "foreclosure_id"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_accelerator_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "accelerator"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_tax_id_accelerator_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "tax_id_accelerator"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_property_type_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "property_type"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_size_of_land_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "size_of_land"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_percentage_of_ownership_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "percentage_of_ownership"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_notary_public_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "notary_public"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # add any found entities into a JSON format within collective_dict
#         if len(entities) > 0:
#             results = [text, {"entities": entities}]
#             TRAIN_DATA['TRAINING_DATA'].append(results)
#
#         return TRAIN_DATA
#
#
#     # Create DocBin collection of example objects
#     def create_training(TRAIN_DATA, nlp):
#         db = DocBin()
#         for text, annot in tqdm(TRAIN_DATA):
#             doc = nlp.make_doc(text)
#             ents = []
#
#             # create span objects
#             for start, end, label in annot["entities"]:
#                 span = doc.char_span(start, end, label=label, alignment_mode="contract")
#
#                 # skip if the character indices do not map to a valid span
#                 if span is None:
#                     print("Skipping entity.")
#                 else:
#                     ents.append(span)
#                     # handle erroneous entity annotations by removing them
#                     try:
#                         doc.ents = ents
#                     except:
#                         # print("BAD SPAN:", span, "\n")
#                         ents.pop()
#             doc.ents = ents
#
#             # pack Doc objects into DocBin
#             db.add(doc)
#             # Export results (here I add it to a TRAIN_DATA folder within the directory)
#             db.to_disk("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/train_data/train_data.spacy")
#
#
#     def structure_dev_data(text, kw_auction_date_list, kw_tax_id_of_debtor_list, kw_auction_id_list, kw_eauction_website_list,
#                                 kw_foreclosure_id_list, kw_accelerator_list, kw_tax_id_accelerator_list, kw_property_type_list,
#                                 kw_size_of_land_list, kw_percentage_of_ownership_list, kw_notary_public_list, txt_filepaths):
#
#         # This dictionary will contain all annotated examples for Dev, which is reserved for the validation dataset
#         collective_dict_2 = {'DEV_DATA': []}
#         entities = []
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_auction_date_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "auction_date"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_tax_id_of_debtor_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "tax_id_of_debtor"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_auction_id_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "auction_id"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_eauction_website_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "eauction_website"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_foreclosure_id_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "foreclosure_id"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_accelerator_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "accelerator"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_tax_id_accelerator_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "tax_id_accelerator"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_property_type_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "property_type"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_size_of_land_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "size_of_land"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_percentage_of_ownership_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "percentage_of_ownership"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # search for instances of keywords within the text (ignoring letter case)
#         for kw in tqdm(kw_notary_public_list):
#             search = re.finditer(kw, text, flags=re.IGNORECASE)
#
#             # store the start/end character positions
#             all_instances = [[m.start(), m.end()] for m in search]
#
#             # if the callable_iterator found matches, create an 'entities' list
#             if len(all_instances) > 0:
#                 for i in all_instances:
#                     start = i[0]
#                     end = i[1]
#                     entities.append((start, end, "notary_public"))
#
#             # alert when no matches are found given the user inputs
#             else:
#                 print("No pattern matches found. Keyword:", kw)
#
#         # add any found entities into a JSON format within collective_dict_2
#         if len(entities) > 0:
#             results = [text, {"entities": entities}]
#             collective_dict_2['DEV_DATA'].append(results)
#
#         # Loop txt files to use as dev data (validation data)
#         for i in txt_filepaths[101:130]:
#             with open(i, 'r', encoding="utf8") as f:
#                 f.read().replace('\n', ' ')
#
#
#         # define our training data to DEV_DATA
#         DEV_DATA = collective_dict_2['DEV_DATA']
#
#         # create a blank model
#         nlp = spacy.blank("xx")
#         return nlp, DEV_DATA
#
#
#     def create_dev(DEV_DATA, nlp):
#         db = DocBin()
#         for text, annot in tqdm(DEV_DATA):
#             doc = nlp.make_doc(text)
#             ents = []
#
#             # create span objects
#             for start, end, label in annot["entities"]:
#                 span = doc.char_span(start, end, label=label, alignment_mode="contract")
#
#                 # skip if the character indices do not map to a valid span
#                 if span is None:
#                     print("Skipping entity.")
#                 else:
#                     ents.append(span)
#                     # handle erroneous entity annotations by removing them
#                     try:
#                         doc.ents = ents
#                     except:
#                         # print("BAD SPAN:", span, "\n")
#                         ents.pop()
#             doc.ents = ents
#
#             # pack Doc objects into DocBin
#             db.add(doc)
#             # Export results (here I add it to a TRAIN_DATA folder within the directory).
#             db.to_disk("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/dev_data/dev_data.spacy")
#
#     def test_trained_model(self):
#         """
#         After creating train and dev data sets written in the code above, follow the steps below:
#
#         1. Follow the link: https://spacy.io/usage/training#config and select ‘ner’ in the Components section of the widget.
#         Copy/paste the full contents of the widget into a file named base_config.cfg within your folder directory.
#
#         2. Open the base_config.cfg file. All you need to change is the train and dev variables at the top.
#         Dev is reserved for the validation dataset. However, if you just want to train the model for demonstration purposes,
#         add the same path to both train and dev. (Note: this will result in 100% training accuracy)
#
#         3. Open your CLI (terminal) and cd over into the directory of base_config.cfg and simply copy/paste this command:
#         python -m spacy init fill-config base_config.cfg config.cfg
#         A config.cfg file will appear in your working directory.
#
#         4. Run the following to begin training:
#         python -m spacy train config.cfg --output C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/results
#         After training is complete, the resulting model will appear in a new folder called output.
#
#         Your model is trained. Proceed below.
#         """
#
#         # Test out the model we trained on a new text file, created from auction pdfs.
#         with open(r'C:\Users\kostas.skepetaris\PycharmProjects\web_retrieve_data\static\texts\report166028.txt', 'r', encoding="utf8") as f:
#             test_1 = f.read().replace('\n', ' ')
#
#         # Load the trained model
#         nlp_output = spacy.load("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/results/model-best")
#
#         # Pass our test file into the trained pipeline.
#         doc = nlp_output(test_1)
#
#         # Print out the identified entities found in test text file.
#         for ent in doc.ents:
#             if ent.label_ == "auction_date":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "tax_id_of_debtor":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "auction_id":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "eauction_website":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "foreclosure_id":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "accelerator":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "tax_id_accelerator":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "property_type":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "size_of_land":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "percentage_of_ownership":
#                 print(ent.text, ent.label_)
#         for ent in doc.ents:
#             if ent.label_ == "notary_public":
#                 print(ent.text, ent.label_)


def annotatedEntitesTxtFilepaths():

    # Doccano annotation tool
    # https://github.com/doccano/doccano
    # When you run doccano add labels as span
    all_annotated_entities_as_json = pd.read_json(path_or_buf=r'C:\Users\kostas.skepetaris\Downloads\all.jsonl',
                                                  lines=True)

    '''I found the issue with Django here: https://code.djangoproject.com/wiki/JSON1Extension. 
    My Python version (3.8.10) does not include the SQLLite JSON1 extension by default. Switching out the sqlite DLLs fixed the issue.'''

    # With the help of: https://levelup.gitconnected.com/auto-detect-anything-with-custom-named-entity-recognition-ner-c89d6562e8e9

    return all_annotated_entities_as_json


def create_training_and_dev(all_annotated_entities_as_json):
    '''First, we need to convert our training examples from JSON format into spaCy Doc objects.
    The Doc objects are then stored in a collective DocBin class. This is just the new format that spaCy requires for training.'''
    # create a blank model
    nlp = spacy.blank("xx")
    # Create DocBin collection of example objects
    db = DocBin()
    training_data = []
    test_data = []
    annotated_dataset = all_annotated_entities_as_json.values
    np.random.shuffle(annotated_dataset)
    split_horizontally_idx = int(annotated_dataset.shape[0]* 0.8)
    training = annotated_dataset[:split_horizontally_idx, :]
    test = annotated_dataset[split_horizontally_idx:, :]
    for id in training:
        training_data.append((id[1], id[3]))

    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        ents = []

        # create span objects
        for start, end, label in annotations:
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
        # Export results (here I add it to a TRAIN_DATA folder within the directory)
        db.to_disk("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/train_data/train_data.spacy")

    db = DocBin()
    for id in test:
        test_data.append((id[1], id[3]))
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        ents = []

        # create span objects
        for start, end, label in annotations:
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
        # Export results (here I add it to a TRAIN_DATA folder within the directory).
        db.to_disk("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/dev_data/dev_data.spacy")


def extract_labels_from_files():
    """
    After creating train and dev data sets written in the code above, follow the steps below:

    1. Follow the link: https://spacy.io/usage/training#config and select ‘ner’ in the Components section of the widget.
    Copy/paste the full contents of the widget into a file named base_config.cfg within your folder directory.

    2. Open the base_config.cfg file. All you need to change is the train and dev variables at the top.
    Dev is reserved for the validation dataset. However, if you just want to train the model for demonstration purposes,
    add the same path to both train and dev. (Note: this will result in 100% training accuracy)

    3. Open your CLI (terminal) and cd over into the directory of base_config.cfg and simply copy/paste this command:
    python -m spacy init fill-config base_config.cfg config.cfg
    A config.cfg file will appear in your working directory.

    4. Run the following to begin training:
    python -m spacy train config.cfg --output C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/results
    After training is complete, the resulting model will appear in a new folder called output.

    Your model is trained. Proceed below.
    """

    # Test out the model we trained on a new text file, created from auction pdfs.
    with open(r'C:\Users\kostas.skepetaris\PycharmProjects\web_retrieve_data\static\texts\report166028.txt', 'r', encoding="utf8") as f:
        test_1 = f.read().replace('\n', ' ')

    # Load the trained model
    nlp_output = spacy.load("C:/Users/kostas.skepetaris/PycharmProjects/object_detection_nlp/app/static/results/model-best")

    # Pass our test file into the trained pipeline.
    doc = nlp_output(test_1)

    # Print out the identified entities found in test text file.
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


if __name__ == '__main__':
    all_annotated_entities_as_json = annotatedEntitesTxtFilepaths()
    training = create_training_and_dev(all_annotated_entities_as_json)
    test = extract_labels_from_files()

