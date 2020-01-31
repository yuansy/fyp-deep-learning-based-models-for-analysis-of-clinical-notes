"""extracting note and metadata from mimic csv; train word embeddings"""

import pandas as pd
import HP
from preprocess_utilities import train_embedding, load_embedding, preprocess, split_doc, tokenize


# read csv file from mimic csv files
df_note = pd.read_csv(HP.mimic_note_events)
admission = pd.read_csv(HP.mimic_admissions) # admission table
patient = pd.read_csv(HP.mimic_patients)  # patient demographic information


# get the usefule dataframe
patient_note_label, patient_subjectid2index = preprocess(df_note, admission, patient)
# save the patient subject_id_2_row_index to a csv file
patient_subjectid2index.to_csv(HP.subject_index)


# train embeddings
train_embedding(patient_note_label)
# get the pre-trained embeddings from mimic
mimic3_embedding = load_embedding()


# extract text file for prediction model
result = open(HP.result_csv, "w")
result.write("patient_id,dead_after_disch_date\n")
for index, row in patient_note_label.iterrows():
    
    # if extract label, uncomment the following section:
    tmp_full_text = row['full_text']
    tmp_dead_after_disch_date = row["dead_after_disch_date"]
    tmp_patient_id = "patient" + str(index)
    result.write(tmp_patient_id + "," + str(tmp_dead_after_disch_date) + "\n")

    # if extract notes, uncomment the following section: 
    f = open(HP.data_directory + tmp_patient_id + '.txt', 'w')
    for x in row['full_text']:
        category = x[0].strip()
        category_index = HP.category_id[category]
        doc = x[1]
        sentences = split_doc(doc)
        for sent in sentences:
            cleaned_tokens = tokenize(sent, mimic3_embedding)
            if len(cleaned_tokens) > 0:
                f.write(str(category_index) + "\n")
                for t in cleaned_tokens:
                    f.write(t + "\n")
                f.write("\n")
    f.close() 

result.close()