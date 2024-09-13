import ultralytics
import cv2
from utils import read_video, save_video
from trackers import Tracker
from team_color_assigner import ColorAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    video_frames = read_video('input_video/08fd33_4.mp4')
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_track(video_frames, read_from_stub= True, stub_path= 'stubs/tracks_stubs.pkl')
    
    # Get missing value of ball's position
    tracks["ball"] = tracker.get_ball_position(tracks['ball'])
    # Assign team color to player
    color_assigner = ColorAssigner()
    color_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = color_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color']= color_assigner.team_color[team]
    
    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball']= True
    
    output = tracker.draw_annotation(video_frames, tracks)
    save_video(output, 'output_video/video.mp4')
    

if __name__ == '__main__':
    main()