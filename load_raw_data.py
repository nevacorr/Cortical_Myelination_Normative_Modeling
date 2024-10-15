import pandas as pd
import numpy as np
def load_raw_data(struct_var, visit, datafilename):
    # Load MPF, cortical thickness gm, affective behavior and social media data
    genz_data_combined = pd.read_csv(datafilename)

    # remove rows with nans for gender
    genz_data_combined = genz_data_combined.dropna(subset=['gender'])

    # convert gender, agegroup and agemonths columns from float to int
    genz_data_combined['gender'] = genz_data_combined['gender'].astype('int64')
    genz_data_combined['agegroup'] = genz_data_combined['agegroup'].astype('int64')
    genz_data_combined['agemonths'] = genz_data_combined['agemonths'].astype('int64')
    genz_data_combined['agedays'] = genz_data_combined['agedays'].astype('int64')

    ##########
    # Load data quality measures and puberty values
    ##########
    if visit == 1:
        mpf_quality = pd.read_csv('/home/toddr/neva/genz/mpf_data_quality_visit1.csv')
        euler = pd.read_csv('/home/toddr/neva/genz/freesurfer_processing/visit1_v7/visit1_euler_numbers_18Sep2023',
                            header=None)
        pds = pd.read_csv(
            '/home/toddr/neva/PycharmProjects/Puberty_Analysis/GenZ Puberty Scaling Visit 1 for python.csv')
    elif visit == 2:
        mpf_quality = pd.read_csv('/home/toddr/neva/genz/mpf_data_quality_visit2.csv')
        euler = pd.read_csv('/home/toddr/neva/genz/freesurfer_processing/visit2_v7/visit2_euler_numbers_18Sep2023',
                            header=None)
        pds = pd.read_csv(
            '/home/toddr/neva/PycharmProjects/Puberty_Analysis/GenZ Puberty Scaling Visit 2 for python.csv')

    pds.rename(columns={'Subject ID': 'subject', 'Puberty Development Scale Scoring': 'puberty'}, inplace=True)

    ##########
    # Average left and right hemisphere euler numbers
    ##########
    euler['euler'] = (euler.iloc[:, 1] + euler.iloc[:, 2]) / 2.0
    euler.drop(euler.columns[[1, 2]], axis=1, inplace=True)
    euler['visit'] = visit
    euler['euler'] = euler['euler'].astype(int)
    # calculate median euler value
    median_euler = euler['euler'].median()
    # subtract median euler from all subjects, then multiply by -1 and take the square root
    euler['euler'] = euler['euler'] - median_euler
    euler['euler'] = euler['euler'] * -1
    euler['euler'] = np.sqrt(np.absolute(euler['euler']))

    ##########
    # Insert data quality measure and puberty value into dataframe with brain and behavior data
    ##########
    euler.rename(columns={0: "subject"}, inplace=True)
    pds.rename(columns={"Subject": "subject", "Puberty Stage (PDS)": "puberty"}, inplace=True)
    mpf_quality.rename(columns={"Subj": "subject", "score": "mpf_qscore"}, inplace=True)
    pds['visit'] = visit
    mpf_quality['visit'] = visit
    genz_data_combined = genz_data_combined.merge(euler, how='left', on=["subject", "visit"])
    genz_data_combined = genz_data_combined.merge(pds, how='left', on=["subject", "visit"])
    genz_data_combined = genz_data_combined.merge(mpf_quality, how='left', on=["subject", "visit"])
    #remove first column that has nodata
    genz_data_combined.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    # if struct_var is equal to gmv or cortthick exclude any rows where already transformed euler value is greater than or equal to 10
    if struct_var == 'gmv' or struct_var == 'cortthick':
        keeprows = (genz_data_combined['euler'] < 10.00000) | (genz_data_combined['euler'].isna())
        genz_data_combined = genz_data_combined.loc[keeprows, :]
    elif struct_var == 'mpf':
        keeprows = (genz_data_combined['mpf_qscore'] > 1) | (genz_data_combined['mpf_qscore'].isna())
        genz_data_combined = genz_data_combined.loc[keeprows, :]

    ##########
    # Prepare covariate data
    # E########
    cov = pd.DataFrame()
    cov['participant_id'] = genz_data_combined['subject']
    cov['age'] = genz_data_combined['agegroup']
    cov['visit'] = genz_data_combined['visit']
    cov['sex'] = genz_data_combined['gender']
    cov['euler'] = genz_data_combined['euler']
    cov['puberty'] = genz_data_combined['puberty']
    cov['mpf_qscore'] = genz_data_combined['mpf_qscore']

    return cov, genz_data_combined
