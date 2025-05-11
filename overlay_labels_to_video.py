"""
This script provides the function to overlay predicted labels on videos.


@author: zhang

"""
import os
os.chdir('D:/P/Pscript')


import pandas as pd
import cv2


def add_labels_to_video(input_video_path,
                        output_video_path,
                        df_predicted_label,
                        text_loc=(50, 50), 
                        font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=1,
                        color=(0, 255, 0), # green
                        thickness=2,
                        ):
    
    '''
    overlay labels on each frame.
    
    Args:
        input_video_path (str): Full Path to open the input video.
        output_video_path (str): Path with file name to save the output video.
        df_predicted_label (dataframe): A dataframe that have two columns, frame index (int) and predicted label (str).
        font (int): OpenCV font type (default: cv2.FONT_HERSHEY_SIMPLEX).
        font_scale (float): Font size (default: 1).
        color (tuple): Text color in RGB (default: green).
        thickness (int): Text thickness (default: 2)
        
        
    example of using:
        
    add_labels_to_video(
        input_path,
        output_path,
        df
        )
    '''
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total number of frames: {total_frames}")
    
    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    
    df = df_predicted_label
    
    while frame_idx < len(df): 

        ret, frame = cap.read()
    
        if not ret:
            break
    
        # Get the label for the current frame from the DataFrame
        label = df.loc[df['frame_idx'] == frame_idx, 'pred_motif'].values
    
        if len(label) > 0:
            label_text = str(label[0])  # Get the label value
            cv2.putText(frame, label_text, text_loc, font, font_scale, color, thickness, cv2.LINE_AA)
    
        # Write the frame with label to the output video
        out.write(frame)
    
        frame_idx += 1
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('video labelling successed')
    
    