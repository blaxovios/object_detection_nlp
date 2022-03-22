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

# With the help of: https://levelup.gitconnected.com/auto-detect-anything-with-custom-named-entity-recognition-ner-c89d6562e8e9

text_filepath = 'C:/Users/kostas.skepetaris/PycharmProjects/web_retrieve_data/static/results/deltio_nomikon/deltio 08-03-2022.xlsx'
text_df = pd.read_excel(text_filepath)
labels_df = pd.read_excel('C:/Users/kostas.skepetaris/PycharmProjects/web_retrieve_data/static/csv/Weekly update 11012022_17022022.xlsx')
labels_list = list(labels_df.columns)
text_df = text_df.reindex(columns=[*text_df.columns.tolist(), 'labels'], fill_value=str(labels_list))
text_description_labels_df = text_df[['Asset Description', 'labels']]
text_description_labels_df = text_description_labels_df[~text_description_labels_df['Asset Description'].duplicated()]

# this dictionary will contain all annotated examples
collective_dict = {'TRAINING_DATA': []}


def structure_training_data(text, kw_list):
	results = []
	entities = []

	# search for instances of keywords within the text (ignoring letter case)
	for kw in tqdm(kw_list):
		search = re.finditer(kw, text, flags=re.IGNORECASE)

		# store the start/end character positions
		all_instances = [[m.start(), m.end()] for m in search]

		# if the callable_iterator found matches, create an 'entities' list
		if len(all_instances) > 0:
			for i in all_instances:
				start = i[0]
				end = i[1]
				entities.append((start, end, "SERVICE"))

		# alert when no matches are found given the user inputs
		else:
			print("No pattern matches found. Keyword:", kw)

	# add any found entities into a JSON format within collective_dict
	if len(entities) > 0:
		results = [text, {"entities": entities}]
		collective_dict['TRAINING_DATA'].append(results)
		return