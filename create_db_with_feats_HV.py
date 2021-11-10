from tkinter import *
from tkinter.filedialog import askopenfilename
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#_____________________________________
# Creating 'browse' windows for user to select elemental and hea data files

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
el_filename = askopenfilename(initialdir=os.getcwd(),
        title='Select element database file (.csv)') # show an "Open" dialog box and return the path to the selected file
        
hea_filename = askopenfilename(initialdir=os.getcwd(),
        title='Select hea_data file (.csv)') # show an "Open" dialog box and return the path to the selected file

db_element = pd.read_csv(el_filename,encoding='latin-1')
db_element = db_element.set_index('Symbol') #set 'Symbol' column as index


#_____________________________________
# Taking user inputs

norm = input('Do you want normalized features (y/n)? :  ')
feature_order = int(input('What order features are required? (1/2/3): '))
y_prop = input('Which prop database? (VHN/YS_MPa/UTS_MPa/elongation/Y_mod_GPa): ')
db_save_name = input('Save database as: (___.csv): ')


#_____________________________________
# Importing scripts that will be used 

common_dir = '_scripts'

# setting common_dir as a system path so that scripts can be imported from there
sys.path.insert(1, common_dir) # insert at 1, DO NOT INSERT AT 0: 0 is the script path (or '' in REPL)

from add_db_features import add_feats_to_db


#_____________________________________
# Creating database with features
db_hea = pd.read_csv(hea_filename,encoding='latin-1'); #reading input hea database
db = add_feats_to_db(db_hea); #creating database with features using 'add_feats_to_db' script


#_____________________________________
# Keeping only relevant data in the dataset

y_prop = [y_prop]
feats = ['R_delta','S_config','VEC','R_cov_delta','density_avg','Tm_avg','E_avg','E_delta',
        'G_avg','G_delta','B_avg','B_delta','Compress_avg','Compress_delta','EN_Allen_avg',
        'H_avg','Vm_delta','E_coh_avg','E_coh_delta','Senkov_param','H_ch_M_L_R','H_el_M_S_R']
        

db = db[['alloy_name', 'phases'] + y_prop + feats]; #storing only required data
db = db.dropna(axis='index', subset = y_prop + feats); #dropping nan values

print('Dataset shape (after removing nan values):',db.shape)
db.reset_index(drop=True, inplace=True)


#_____________________________________
# Normalizing (if user selected it)

Y = db[db.columns[0:3]]; #creating a Y df -> this will NOT be normalized
X = db[db.columns[3:]]; #creating a X df -> this will be normalized

if norm == 'n':
    X_new = X

else:
    xmin = np.amin(X, axis=0); #storing min of each column
    xmax = np.amax(X, axis=0); #storing max of each column

    xmin.to_csv('xmin_'+db_save_name+'.csv'); #saving x_min values
    xmax.to_csv('xmax_'+db_save_name+'.csv'); #saving x_max values

    X_new = (X-xmin)/(xmax-xmin) #min-max normalization


#_____________________________________
# Creating polynomial features and creating final database
poly = PolynomialFeatures(feature_order)
new = poly.fit_transform(X_new)
poly_labels = poly.get_feature_names(X_new.columns)
len(poly_labels)
    
new_X = pd.DataFrame(new, columns= poly_labels)
new_X = new_X.drop(columns='1')
    
db_comb = pd.concat([Y, new_X], axis=1, sort=False)


#_____________________________________
# Saving final database and min_max values
db_comb.to_csv(db_save_name+'.csv'); #saving normalized final database

print(str(db_save_name)+'.csv saved.')
