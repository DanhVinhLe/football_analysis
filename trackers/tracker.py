import supervision as sv
import cv2
from ultralytics import YOLO
import pickle
import os
from utils.bbox import get_box_width, get_center_of_box, get_foot_position
import numpy as np 
import pandas as pd

class Tracker:
    def __init__(self, model_path): 
        super().__init__()
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def get_ball_position(self, ball_position):
        """_summary_
        Get missing value when detect ball
        Args:
            ball_position (list): tracks['ball'] in dictionary tracks
        """
        ball_positions = [x.get(1, {}).get('bbox',[]) for x in ball_position]
        df_ball_positions = pd.DataFrame(ball_positions, columns= ['x1', 'y1', 'x2', 'y2'])
        
        # Get missing value
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions 
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range (0, len(frames), batch_size):
            detect = self.model.predict(frames[i: i+batch_size], conf = 0.1)
            detections += detect
        return detections

    def get_object_track(self, frames, read_from_stub = False, stub_path = None):
        """_summary_

        Args:
            frames : input video frames
            read_from_stub (bool, optional): if true, return tracks from path, don't need to run from scratch, default is false
            stub_path (string, optional): path to read or write tracks, default is true

        Returns:
            _type_: _description_
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks = {
                'players': [],
                'referee': [],
                'ball': []
            }
        '''
        For example:
        tracks['player'] is a list of dictionary, where each dictionary is a pair of id of player and his bbox, position
        and the ith dictionary is corresponding to ith frame
        tracks['player'] = [{1: {'bbox':[0,0,0,0]}, 2: {'bbox':[0,0,1,1]}}, -> 1st frame
                            ....}]
        tracks will be update with some attributes like 'team' and 'team_color'
        the same with referee and ball 
        '''
        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v:k for k,v in class_names.items()}
            
            # convert from YOLO format to supervision format 
            detections_sv = sv.Detections.from_ultralytics(detection)
            
            
            # convert goalkeeper to player 
            for obj_ind, class_id in enumerate(detections_sv.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detections_sv.class_id[obj_ind] = class_names_inv['player']
            # Track objects
            detection_with_track = self.tracker.update_with_detections(detections_sv)
            tracks['players'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})
            # loop each object with track
            for frame_detect in detection_with_track:
                bbox = frame_detect[0].tolist()
                cls_id = frame_detect[3]
                track_id = frame_detect[4]
                if cls_id == class_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox": bbox, "position": get_foot_position(bbox)}
                if cls_id == class_names_inv['referee']:
                    tracks['referee'][frame_num][track_id] = {"bbox": bbox, "position": get_foot_position(bbox)}
            for frame_detect in detections_sv:
                bbox = frame_detect[0].tolist()
                cls_id = frame_detect[3]
                
                if cls_id == class_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox": bbox}
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks
    
    def draw_ellipse(self, frame, bbox, color):
        """_summary_

        Args:
            frame (matrix): current frame
            bbox : bounding box of object
            color : color in RGB
        """
        y2 = int(bbox[3])
        x_center, y_center = get_center_of_box(bbox)
        width = get_box_width(bbox)
        
        cv2.ellipse(frame, center=(x_center, y2), axes= (int(width), int(0.35*width)),
                    angle= 0.0, startAngle= -45, endAngle=235, color= color,
                    thickness= 2, lineType= cv2.LINE_4)
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y1 = int(bbox[1])
        x, _ = get_center_of_box(bbox)
        triangle_points = np.array([[x,y1], [x-10, y1-20], [x+10, y1-20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 0)
        return frame
    
    def draw_annotation(self, video_frames, tracks):
        """_summary_

        Args:
            video_frames : list of input frames
            tracks : tracks of object 
        """
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referee'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            for _, player in player_dict.items():
                color = player.get("team_color", (0,0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color)
                
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0, 255))
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255,255))
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (255,0, 0))
            output_video_frames.append(frame)
        return output_video_frames
        
        