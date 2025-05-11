"""
This script provides helper functions to acquire motion features.

@author: zhang

"""
import numpy as np
import pandas as pd
from scipy.spatial import distance



def cal_distance(Dataframe, DLCscorer, bpt):

    """
    Calculates distance from (0,0) and XY position array for a body part.
    
    Args:
        DLCscorer (str): Name of the DLC scorer to use from the Dataframe.
        bpt (str): Name of the body part to analyze.
        
    Returns:
        - dist (numpy.ndarray): Array of distances from (0,0), baseline-corrected
        - pos_arr (numpy.ndarray): 2D array of XY positions (Nx2 array)
            
    Note:
        The distance values are baseline-corrected by subtracting the mean distance.
    """

    x = Dataframe[DLCscorer][bpt]['x'].values
    y = Dataframe[DLCscorer][bpt]['y'].values
    dist = np.sqrt(np.square(x) + np.square(y)) # distance to the (0,0)
    dist = dist - np.mean(dist)                 # correct the baseline
    pos_arr = np.transpose(np.vstack((x,y)))    # prepare array for position XY
    return dist, pos_arr


def cal_speed(pos_arr):
    """
    Calculates speed from XY position data.
    
    Args:
        pos_arr (numpy.ndarray): 2D array of XY positions (Nx2 array)
        
    Returns:
        numpy.ndarray: Array of speed magnitudes (in pixels/time step)
        
    Note:
        - speed is calculated as the Euclidean distance between consecutive points
        - First frame speed is always 0
        - Returns pixel speed (not normalized by time between frames)
    """

    vel = []
    for n,pos in enumerate(pos_arr):
        if n==0:# for the first frame, initialize the position and add 0 as the speed
            p0 = pos
            vel.append(0)
        else:
            p1 = pos
            v = np.abs(distance.euclidean(p0,p1))
            vel.append(v)
            p0 = p1
    return np.array(vel)


def cal_acceleration(speedArr):

    """
    
    Calculates instantaneous acceleration from a time series of speed values.
    
    Args:
        speedArr (numpy.ndarray): 1D array containing sequential speed measurements.
                               Units should be in pixels per time step.

    Returns:
        numpy.ndarray: 1D array of acceleration values with same length as input.
                      Units are in pixels per time step squared (px/t²).
                      First value is always 0 (no acceleration at initial time point).
                      
    Note:
        - Acceleration is calculated as the finite difference between consecutive speeds: a = Δv/Δt
        - Since this assumes uniform time steps (Δt=1), acceleration simplifies to a = v[n] - v[n-1]
        - For smoother results, consider applying a moving average to the input velocities first
        

    """
    acc = []
    for n, vel in enumerate(speedArr):
        if n == 0:
            v0 = vel
            acc.append(0)
        else:
            v1 = vel
            a = v1 - v0
            acc.append(a)
            v0 = v1
    return np.array(acc)





#%% angle features

import math

def calculate_angle_3points(point_a, point_b, point_c):
    
    """
    Calculates the angle formed by three points (∠BAC) in degrees.

    Computes the angle at point A between vectors AB and AC using the dot product formula:
    θ = arccos((AB · AC) / (|AB| * |AC|))

    Args:
        point_a (numpy.ndarray): (x,y) coordinates of vertex point A
        point_b (numpy.ndarray): (x,y) coordinates of point B
        point_c (numpy.ndarray): (x,y) coordinates of point C

    Returns:
        float: Angle in degrees (0-180°) formed by points A, B, and C (∠BAC)

    Raises:
        ValueError: If any two points are coincident (would create zero-length vector)
        ArithmeticError: If floating point error occurs in arccos calculation

    Notes:
        - The angle is always the smallest angle between the two vectors (0°-180°)
        - Points should be provided in Cartesian coordinates (x,y)

    Example:
        >>> calculate_angle_3points((0,0), (1,0), (0,1))
        90.0

    """
    # Calculate vectors AB and AC
    vector_ab = [point_b[0] - point_a[0], point_b[1] - point_a[1]]
    vector_ac = [point_c[0] - point_a[0], point_c[1] - point_a[1]]

    # Calculate dot product of AB and AC
    dot_product = vector_ab[0] * vector_ac[0] + vector_ab[1] * vector_ac[1]

    # Calculate magnitudes of AB and AC
    magnitude_ab = math.sqrt(vector_ab[0]**2 + vector_ab[1]**2)
    magnitude_ac = math.sqrt(vector_ac[0]**2 + vector_ac[1]**2)

    # Handle potential division by zero from coincident points
    if magnitude_ab == 0 or magnitude_ac == 0:
        raise ValueError("Two or more points are coincident, cannot form angle")

    # Calculate the cosine of the angle between AB and AC
    cosine_angle = dot_product / (magnitude_ab * magnitude_ac)

    # Handle potential floating point errors
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    # Calculate the angle in radians using arccosine
    angle_radians = math.acos(cosine_angle)

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def get_joint_angle (Dataframe, DLCscorer, legPos):
    '''
    Calculates joint angles over time for a specified leg using three anatomical points.    
    Args:
        Dataframe (pd.DataFrame): DeepLabCut output dataframe containing tracking data
        DLCscorer (str): Name of the DLC scorer/model to use in the dataframe
        legPos (str): Leg position identifier (e.g., 'left front', 'right rear'). 
                     Used to construct body part names in the format:
                     legPos + ' tibia-tarsus joint' (point A)
                     legPos + ' femur-tibia joint' (point B) 
                     legPos + ' leg tip' (point C)
    Returns:
    angle(numpy.ndarray): Array of joint angles in degrees for each frame (0°-180°),
    
    Example of using: 
    >>> legPos = 'left front'
    >>> angleTTJ_lf = get_joint_angle(Dataframe, DLCscorer, legPos)
    
    '''
    bpt = legPos + ' tibia-tarsus joint'
    x = Dataframe[DLCscorer][bpt]['x'].values
    y = Dataframe[DLCscorer][bpt]['y'].values
    pos_a = np.transpose(np.vstack((x,y)))
    
    bpt = legPos + ' femur-tibia joint'
    x = Dataframe[DLCscorer][bpt]['x'].values
    y = Dataframe[DLCscorer][bpt]['y'].values
    pos_b = np.transpose(np.vstack((x,y)))
    
    bpt = legPos + ' leg tip'
    x = Dataframe[DLCscorer][bpt]['x'].values
    y = Dataframe[DLCscorer][bpt]['y'].values
    pos_c = np.transpose(np.vstack((x,y)))
    
    angle = np.zeros(len(pos_a))
    
    for n in range(len(pos_a)):
        point_a = pos_a[n,:]
        point_b = pos_b[n,:]
        point_c = pos_c[n,:]
        angle[n] = calculate_angle_3points(point_a, point_b, point_c)

    return angle


# angle Speedocity

def calculate_angSpeed(angle):
    
    '''
    Calculates the angular speed (first derivative of joint angle) over time.
    
    Computes the finite difference between consecutive joint angle measurements to estimate
    instantaneous angular speed in degrees per frame. The first frame speed is always 0.
    
    Args:
        angle (numpy.ndarray): 1D array of joint angles in degrees (0-180°), typically from
                             `get_joint_angle()` or similar function. Shape (n_frames,).
    
    Returns:
        numpy.ndarray: Array of angular speeds in degrees/frame. Shape (n_frames,)
    
    
    Example:
        >>> angSpeedTTJ_lf = calculate_angSpeed(angleTTJ_lf)
    
    '''

    angSpeed = []
    for n,ang in enumerate(angle):
        if n == 0:
            ang0 = ang
            angSpeed.append(0)
        else:
            ang1 = ang
            v_ang = ang1 - ang0
            angSpeed.append(v_ang)
            ang0 = ang1
    return np.array(angSpeed)


#%% leg-body angle

def calculate_angle_ByVectors(vector_u, vector_v):
    '''
    Calculates the signed angle between two 2D vectors in degrees (0-360°).
    Args:
        vector_u (numpy.ndarray): First 2D vector [x,y]
        vector_v (numpy.ndarray): Second 2D vector [x,y]
    
    Returns:
        float: Angle between vectors in degrees (0-360°)
        
    '''
    
    dot_product = np.dot(vector_u, vector_v)
    magnitude_u = np.linalg.norm(vector_u)
    magnitude_v = np.linalg.norm(vector_v)

    # Ensure denominators are not zero
    if magnitude_u == 0 or magnitude_v == 0:
        return None

    cosine_theta = dot_product / (magnitude_u * magnitude_v)
    angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

    # Convert radians to degrees
    angle_deg = np.degrees(angle_rad)

    # Determine the sign of the angle based on the cross product
    cross_product = np.cross(vector_u, vector_v)
    if cross_product < 0:
        angle_deg = 360 - angle_deg

    return angle_deg



def get_body_leg_angle (Dataframe, DLCscorer, legPos):
    '''
    Calculates the angle between a leg vector and body axis vector over time.
    The angle is computed between:
        - Leg vector: From leg base to femur-tibia joint
        - Body axis vector: From abdomen tip to neck/head
    
    Args:
        Dataframe (pd.DataFrame): DeepLabCut output dataframe containing tracking data
        DLCscorer (str): Name of the DLC scorer/model to use in the dataframe
        legPos (str): Leg position identifier (e.g., 'left front', 'right rear').
    
    Returns:
        numpy.ndarray: Array of angles in degrees (0-360°) for each frame.
        
    e.g.: 
        legPos = 'left front'
        legBodyAngle_lf = get_body_leg_angle(Dataframe, DLCscorer, legPos)

    '''
    bpt = legPos + ' leg base'
    x = Dataframe[DLCscorer][bpt]['x'].values
    y = Dataframe[DLCscorer][bpt]['y'].values
    pos_a = np.transpose(np.vstack((x,y)))
    
    bpt = legPos + ' femur-tibia joint'
    x = Dataframe[DLCscorer][bpt]['x'].values
    y = Dataframe[DLCscorer][bpt]['y'].values
    pos_b = np.transpose(np.vstack((x,y)))
    
    bpt = 'tip of abdomen'
    x = Dataframe[DLCscorer][bpt]['x'].values
    y = Dataframe[DLCscorer][bpt]['y'].values
    pos_c = np.transpose(np.vstack((x,y)))
    
    bpt = 'neck' # or use antennae midpoint
    x = Dataframe[DLCscorer][bpt]['x'].values
    y = Dataframe[DLCscorer][bpt]['y'].values
    pos_d = np.transpose(np.vstack((x,y)))
        
    angle = []
    
    for n in range(len(pos_a)):
        vector_u = pos_b[n,:] - pos_a[n,:]  # vector from leg base to FT joint.
        vector_v = pos_d[n,:] - pos_c[n,:]  # vector from abdomen tip to neck.
        ang = calculate_angle_ByVectors(vector_u, vector_v)
        angle.append(ang)
    return np.array(angle)

 
#%% correlation features

from itertools import combinations

def get_rolling_abs_corr(df1,df2,r_window_size):
    '''
    Calculates rolling absolute Pearson correlation between two time series.

    Computes the correlation in a sliding window, then takes absolute values.
    The window is centered by default (current point at window center).

    Args:
        df1 (pd.Series): First time series
        df2 (pd.Series): Second time series (same length as df1)
        r_window_size (int): Size of the rolling window in samples

    Returns:
        pd.Series: Rolling absolute correlation values (same length as inputs)

    '''

    rolling_r = df1.rolling(window=r_window_size, center=True).corr(df2)
    rolling_r = rolling_r.abs()
    return rolling_r

def get_combinations(n,k):
    '''
    Generates all possible k-length combinations from n elements.

    Args:
        n (int): Total number of elements (range 1 to n)
        k (int): Length of combinations (must be ≤n)

    Returns:
        combinations_list: List of tuples with all combinations
        combination_num: Integer count of combinations (n choose k)

    '''
    elements = range(1, n+1)
    combinations_list = list(combinations(elements, k))
    combination_num = len(combinations_list)
    return combinations_list,combination_num

def get_r_features(df_feature,features_calculate_corr):
    '''
    Computes rolling absolute correlations between all feature pairs.

    Args:
        df_feature (pd.DataFrame): Input dataframe containing time series features
        features_calculate_corr (list): List of column names to correlate
        r_window_size (int, optional): Window size for rolling correlation. Default=50
    
    Returns:
        df_rolling_r_series: DataFrame with rolling correlations (n_samples × n_combinations)
        combs: List of tuples showing which features were paired
        
    '''
    combs, comb_num = get_combinations(len(features_calculate_corr), 2)
    rolling_r = []
    for i in range(comb_num):
        df1 = df_feature[features_calculate_corr].iloc[:,combs[i][0] - 1]
        df2 = df_feature[features_calculate_corr].iloc[:,combs[i][1] - 1]
        r_window_size = 50
        r = get_rolling_abs_corr(df1, df2, r_window_size)
        rolling_r.append(r)
    
    df_rolling_r_series = pd.concat(rolling_r, axis=1) # each coloum is the rolling r of two
                                                        # time series pairs
    return df_rolling_r_series, combs

def get_r_col_names(features_calculate_corr,combs):
    '''
    Generates standardized column names for correlation features.

    Args:
        features_calculate_corr (list): Original feature names
        combs (list): List of combination tuples from get_combinations()

    Returns:
        list: Generated column names in format 'r_prefix_ij' 
        
    '''
    prefix = features_calculate_corr[0].split('_')[0]
    df_r_col_names = []
    for i in range(len(combs)):
        ele_1 = str(combs[i][0])
        ele_2 = str(combs[i][1])
        df_r_col_names.append('r' + '_' + prefix +'_'+ ele_1 + ele_2)
    return df_r_col_names

