import os, json
from write_to_csv import writeToCsv
import pandas as pd

# Put openFDA.json in parent folder
file = os.path.dirname(__file__) + "/../openFDA.json"

with open(file) as project_file:    
    data = json.load(project_file)  

df = pd.json_normalize(data)

#number of medicines used from the database
n = 50

query = [
'brand_name', #openfda
'product_type',
'route',
'generic_name',
'substance_name',
'indications_and_usage',
'dosage_and_administration',
'dosage_forms_and_strengths',
'contraindications',
'warnings_and_cautions',
'adverse_reactions',
'pregnancy',
'nursing_mothers',
'precautions',
'storage_and_handling',
'purpose',
'do_not_use',
'stop_use',
'ask_doctor',
'pregnancy_or_breast_feeding',
'when_using',
'questions',
'information_for_patients'
]

#contains all queries for n medicines
all_inf_dict = {}

for i in range(n):
    inf_dict = {}
    results  = df['results'][0][i]
    result_keys = results.keys()
    print()
    
    for q in query:
        if q == 'product_type' or q == 'route' or q == 'substance_name' or q == 'generic_name' or q == 'brand_name':
            
            if 'openfda' in result_keys:
                if q in results['openfda'].keys():
                    inf_dict[q] = results['openfda'][q][0]
                    
                else: pass
            else: pass
        else:
            
            if q in result_keys:
                inf_dict[q] = results[q][0]
   
    if 'openfda' in result_keys:
        if 'brand_name' in results['openfda'].keys():
            all_inf_dict[results['openfda']['brand_name'][0]] = inf_dict

def getMed(x):
#this is a dict containing all the info about med1
    return all_inf_dict[list(all_inf_dict.keys())[x]]

# medicineList = [getMed(x) for x in range(45)]


# with open('first45.json', 'w') as f:    
#     json.dump(medicineList, f, indent=4)
