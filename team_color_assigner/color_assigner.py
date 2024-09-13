from sklearn.cluster import KMeans

class ColorAssigner():
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}
    
    def get_player_color(self, frame, bbox):
        """_summary_

        Args:
            frame (mat-like): current frame will be processed
            bbox : bounding of the player 

        Returns:
            color of the player
        """
        img = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        top_half_img = img[0: (img.shape[0] //2), :]
        img_2d = top_half_img.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters= 2, init= 'k-means++', n_init= 1)
        kmeans.fit(img_2d)
        
        # get label for each pixel in img
        labels = kmeans.labels_
        
        # Get player cluster
        clustered_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])
        corner = [clustered_img[0][0], clustered_img[0][-1], clustered_img[-1][0], clustered_img[-1][-1]]
        non_player = max(set(corner), key = corner.count)
        player = 1- non_player
        
        player_color = kmeans.cluster_centers_[player]
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        """_summary_

        Args:
            frame : current frame
            player_detections : track['player'][frame_num], player_id: {"bbox": ....}
        """

        # Get color of all player
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
            
        # Use KMeans to get color of each team
        kmeans = KMeans(n_clusters= 2, init= 'k-means++', n_init= 10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        
        self.team_color[1] = kmeans.cluster_centers_[0]
        self.team_color[2] = kmeans.cluster_centers_[1]
        
    def get_player_team(self, frame, player_bbox, player_id):
        """_summary_

        Args:
            frame : current frame
            player_bbox : bounding box of the player
            player_id : player id in current frame, get in tracks

        Returns:
            team_id: team_id of player
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1
        self.player_team_dict[player_id] = team_id
        
        return team_id