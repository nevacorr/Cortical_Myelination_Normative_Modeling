# This function loads all adolescent visit 1 and visit 2 brain structural measures and behavior data,
# demographic measures, puberty measures and MRI data quality measures. It then only keeps the data
# for the visit of interest and returns the structural brain values for all regions, the covariates,
# and a list of all region names
##########

import pandas as pd
from load_raw_data import load_raw_data

def load_genz_data(struct_var, visit, path, datafilename, data_dir):
    cov, genz_data_combined = load_raw_data(struct_var, visit, datafilename, data_dir)

    # remove all rows not from specified visit
    cov = cov.loc[cov['visit'] == visit]

    # drop visit column
    cov.drop(columns='visit', inplace=True)

    # impute missing puberty values
    cov['puberty'] = cov['puberty'].fillna(cov.groupby(['age', 'sex'])['puberty'].transform('mean'))

    ##########
    # Prepare brain data
    ##########
    # make a list of columns of struct variable of interest
    struct_cols = [col for col in genz_data_combined.columns if struct_var + '-' in col]
    # find columns with Cerebellum data
    cerebellum_columns = [col for col in genz_data_combined.columns if "Cerebellum" in col]
    # remove regions with cerebellum from list of struct variables of interest
    struct_cols = [i for i in struct_cols if i not in cerebellum_columns]

    # create brain data dataframe with struct_var columns for visit
    brain_good = pd.DataFrame()
    brain_good['participant_id'] = genz_data_combined['subject']
    brain_good['visit'] = genz_data_combined['visit']
    brain_good['agedays'] = genz_data_combined['agedays']
    print(brain_good.shape)
    print(genz_data_combined.shape)
    if struct_var == 'mpf' or struct_var == 'gmv':
        brain_good[struct_cols] = genz_data_combined[struct_cols]/100.0
    else:
        brain_good[struct_cols] = genz_data_combined[struct_cols]
    # remove all columns with no struct_var values
    brain_good.dropna(inplace=True, axis=0)

    # remove all rows from visit indicated
    brain_good = brain_good.loc[brain_good['visit'] == visit]
    # drop visit column
    brain_good.drop(columns='visit', inplace=True)

    # Check that subject rows align across covariate and brain dataframes
    # Make sure to use how = "inner" so that we only include subjects with data in both dataframes
    all_data = pd.merge(cov, brain_good, how='inner')
    # create a list of all the columns you want to run a normative model for
    roi_ids=all_data.columns.values.tolist()
    #remove subject info from list of brain regions to run normative model on
    subj_info_cols = ['participant_id', 'age', 'sex', 'euler', 'puberty', 'mpf_qscore', 'agedays']
    roi_ids = [r for r in roi_ids if r not in subj_info_cols]


    return brain_good, all_data, roi_ids
