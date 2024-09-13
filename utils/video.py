import cv2

# read video of football match from path
def read_video(video_path):
    '''
    video_path: path of input video to read
    '''
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# save output video

def save_video(output_video_frames, output_video_path):
    '''
    output_video_frames: list of output video frames after analysis
    output_video_path: output path to save video
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc= fourcc, fps = 24.0,
                          frameSize= (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()