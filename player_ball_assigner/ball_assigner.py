from utils import measure_distance, get_foot_position, get_center_of_box

class PlayerBallAssigner():
    def __init__(self):
        self.max_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        """_summary_
        assign ball to player
        Args:
            players : players in current frame
            ball_bbox : bbox of ball in the current frame

        Returns:
            int : id of assigned player, else  -1
        """
        ball_position = get_center_of_box(ball_bbox)
        min_distance = 99999
        
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']
            distance_left = measure_distance(ball_position, (player_bbox[0], player_bbox[-1]))
            distance_right = measure_distance(ball_position, (player_bbox[2], player_bbox[-1]))
            distance = min(distance_left, distance_right)
            
            if distance < self.max_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id
                
        return assigned_player
    