"""
This script is for motion feature generation from DLC outputs. 

@author: zhang

"""

import os
os.chdir(r"D:\P\Pscript\fly paper")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import motion_features_generation_helperFunctions as funcs

data_path = r"D:\Neu\project\7 fly\videos to be ana - fly project\group1_18817_012-12012022160756-0000DLC_resnet152_flies-bang-seizure-MCDec27shuffle4_4300000_filtered.h5" # full path of the DLC output, 'DLC.h5' file.
Dataframe = pd.read_hdf(data_path) 
DLCscorer = Dataframe.columns.get_level_values(0)[0]
bodyparts = Dataframe.columns.get_level_values(1).unique()


# distance, postion, speed, acceleration
bpt = 'left front leg tip'
dist,pos_arr = funcs.cal_distance(Dataframe=Dataframe,DLCscorer=DLCscorer, bpt=bpt)
vel = funcs.cal_speed(pos_arr)
acc = funcs.cal_acceleration(vel)
dist_lf = dist
pos_lf = pos_arr
vel_lf = vel
acc_lf = acc

bpt = 'left middle leg tip'
dist,pos_arr = funcs.cal_distance(Dataframe=Dataframe,DLCscorer=DLCscorer, bpt=bpt)
vel = funcs.cal_speed(pos_arr)
acc = funcs.cal_acceleration(vel)
dist_lm = dist
pos_lm = pos_arr
vel_lm = vel
acc_lm = acc

bpt = 'left rear leg tip'
dist,pos_arr = funcs.cal_distance(Dataframe=Dataframe,DLCscorer=DLCscorer, bpt=bpt)
vel = funcs.cal_speed(pos_arr)
acc = funcs.cal_acceleration(vel)
dist_lr = dist
pos_lr = pos_arr
vel_lr = vel
acc_lr = acc

bpt = 'right front leg tip'
dist,pos_arr = funcs.cal_distance(Dataframe=Dataframe,DLCscorer=DLCscorer, bpt=bpt)
vel = funcs.cal_speed(pos_arr)
acc = funcs.cal_acceleration(vel)
dist_rf = dist
pos_rf = pos_arr
vel_rf = vel
acc_rf = acc

bpt = 'right middle leg tip'
dist,pos_arr = funcs.cal_distance(Dataframe=Dataframe,DLCscorer=DLCscorer, bpt=bpt)
vel = funcs.cal_speed(pos_arr)
acc = funcs.cal_acceleration(vel)
dist_rm = dist
pos_rm = pos_arr
vel_rm = vel
acc_rm = acc

bpt = 'right rear leg tip'
dist,pos_arr = funcs.cal_distance(Dataframe=Dataframe,DLCscorer=DLCscorer, bpt=bpt)
vel = funcs.cal_speed(pos_arr)
acc = funcs.cal_acceleration(vel)
dist_rr = dist
pos_rr = pos_arr
vel_rr = vel
acc_rr = acc


# angleTTJ, angle speed, leg-body angle
legPos = 'left front'
angleTTJ_lf = funcs.get_joint_angle(Dataframe, DLCscorer, legPos)
legPos = 'left middle'
angleTTJ_lm = funcs.get_joint_angle(Dataframe, DLCscorer, legPos)
legPos = 'left rear'
angleTTJ_lr = funcs.get_joint_angle(Dataframe, DLCscorer, legPos)
legPos = 'right front'
angleTTJ_rf = funcs.get_joint_angle(Dataframe, DLCscorer, legPos)
legPos = 'right middle'
angleTTJ_rm = funcs.get_joint_angle(Dataframe, DLCscorer, legPos)
legPos = 'right rear'
angleTTJ_rr = funcs.get_joint_angle(Dataframe, DLCscorer, legPos)

angVelTTJ_lf = funcs.calculate_angSpeed(angleTTJ_lf)
angVelTTJ_lm = funcs.calculate_angSpeed(angleTTJ_lm)
angVelTTJ_lr = funcs.calculate_angSpeed(angleTTJ_lr)
angVelTTJ_rf = funcs.calculate_angSpeed(angleTTJ_rf)
angVelTTJ_rm = funcs.calculate_angSpeed(angleTTJ_rm)
angVelTTJ_rr = funcs.calculate_angSpeed(angleTTJ_rr)


legPos = 'left front'
legBodyAngle_lf = funcs.get_body_leg_angle(Dataframe, DLCscorer, legPos)
legPos = 'left middle'
legBodyAngle_lm = funcs.get_body_leg_angle(Dataframe, DLCscorer, legPos)
legPos = 'left rear'
legBodyAngle_lr = funcs.get_body_leg_angle(Dataframe, DLCscorer, legPos)
legPos = 'right front'
legBodyAngle_rf = funcs.get_body_leg_angle(Dataframe, DLCscorer, legPos)
legPos = 'right middle'
legBodyAngle_rm = funcs.get_body_leg_angle(Dataframe, DLCscorer, legPos)
legPos = 'right rear'
legBodyAngle_rr = funcs.get_body_leg_angle(Dataframe, DLCscorer, legPos)


#%% combine the features to a DataFrame

featureArray = np.vstack((dist_lf, dist_lm, dist_lr, dist_rf, dist_rm, dist_rr,
                          pos_lf[:,0], pos_lm[:,0], pos_lr[:,0], pos_rf[:,0], pos_rm[:,0], pos_rr[:,0], # pos_X
                          pos_lf[:,1], pos_lm[:,1], pos_lr[:,1], pos_rf[:,1], pos_rm[:,1], pos_rr[:,1], # pos_Y
                          vel_lf, vel_lm, vel_lr, vel_rf, vel_rm, vel_rr,
                          acc_lf, acc_lm, acc_lr, acc_rf, acc_rm, acc_rr,
                          angleTTJ_lf, angleTTJ_lm, angleTTJ_lr, angleTTJ_rf, angleTTJ_rm, angleTTJ_rr,
                          angVelTTJ_lf, angVelTTJ_lm, angVelTTJ_lr, angVelTTJ_rf, angVelTTJ_rm, angVelTTJ_rr,
                          legBodyAngle_lf, legBodyAngle_lm, legBodyAngle_lr, legBodyAngle_rf, legBodyAngle_rm, legBodyAngle_rr))

featureArray = featureArray.T

featureName = ["dist_lf", "dist_lm", "dist_lr", "dist_rf", "dist_rm", "dist_rr",
               "posX_lf", "posX_lm", "posX_lr", "posX_rf", "posX_rm", "posX_rr",
               "posY_lf", "posY_lm", "posY_lr", "posY_rf", "posY_rm", "posY_rr",
               "vel_lf", "vel_lm", "vel_lr", "vel_rf", "vel_rm", "vel_rr",
               "acc_lf", "acc_lm", "acc_lr", "acc_rf", "acc_rm", "acc_rr",
               "angleTTJ_lf", "angleTTJ_lm", "angleTTJ_lr", "angleTTJ_rf", "angleTTJ_rm", "angleTTJ_rr",
               "angVelTTJ_lf", "angVelTTJ_lm", "angVelTTJ_lr", "angVelTTJ_rf", "angVelTTJ_rm", "angVelTTJ_rr",
               "legBodyAngle_lf", "legBodyAngle_lm", "legBodyAngle_lr", "legBodyAngle_rf", "legBodyAngle_rm", "legBodyAngle_rr"]

df_feature = pd.DataFrame(featureArray,columns=featureName)

#%% add correlation features 

features_calculate_corr = ["dist_lf", "dist_lm", "dist_lr", "dist_rf", "dist_rm", "dist_rr"]
df_r, combs = funcs.get_r_features(df_feature,features_calculate_corr)
column_names = funcs.get_r_col_names(features_calculate_corr,combs)
df_r.columns = column_names
df_feature = pd.concat([df_feature,df_r], axis=1)

features_calculate_corr = ["posX_lf", "posX_lm", "posX_lr", "posX_rf", "posX_rm", "posX_rr","posY_lf", "posY_lm", "posY_lr", "posY_rf", "posY_rm", "posY_rr"]
df_r, combs = funcs.get_r_features(df_feature,features_calculate_corr)
column_names = funcs.get_r_col_names(features_calculate_corr,combs)
df_r.columns = column_names
df_feature = pd.concat([df_feature,df_r], axis=1)

features_calculate_corr = ["vel_lf", "vel_lm", "vel_lr", "vel_rf", "vel_rm", "vel_rr"]
df_r, combs = funcs.get_r_features(df_feature,features_calculate_corr)
column_names = funcs.get_r_col_names(features_calculate_corr,combs)
df_r.columns = column_names
df_feature = pd.concat([df_feature,df_r], axis=1)

features_calculate_corr = ["acc_lf", "acc_lm", "acc_lr", "acc_rf", "acc_rm", "acc_rr"]
df_r, combs = funcs.get_r_features(df_feature,features_calculate_corr)
column_names = funcs.get_r_col_names(features_calculate_corr,combs)
df_r.columns = column_names
df_feature = pd.concat([df_feature,df_r], axis=1)

# df_feature.to_csv(df_feature_name_to_save, index=False)