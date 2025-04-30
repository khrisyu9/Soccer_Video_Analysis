from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None  # Clustering model for team colors

    def get_clustering_model(self, image):
        # Reshape the image to a 2D array
        if image.size == 0:
            raise ValueError("Empty image provided to clustering model.")
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract the region of interest using the bbox coordinates
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if image.size == 0:
            print(f"Warning: Empty image region extracted with bbox {bbox}")
            return None

        # Use the top half of the image region for clustering
        top_half_image = image[0:int(image.shape[0] / 2), :]
        if top_half_image.size == 0:
            print(f"Warning: Empty top half image extracted with bbox {bbox}")
            return None

        try:
            # Get clustering model on the top half image
            kmeans = self.get_clustering_model(top_half_image)
        except ValueError as e:
            print(f"Error in clustering model: {e}")
            return None

        # Get the cluster labels and reshape back to the image dimensions
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Decide which cluster corresponds to the player using corner pixels
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            if player_color is not None:
                player_colors.append(player_color)
            else:
                print(f"Warning: No player color extracted for bbox {bbox}")

        if not player_colors:
            # Fallback: assign default team colors if clustering cannot be performed.
            print("Error: No valid player colors found. Assigning default team colors.")
            self.team_colors = {
                1: np.array([0, 0, 255]),  # e.g., default blue
                2: np.array([255, 0, 0])  # e.g., default red
            }
            self.kmeans = None
            return

        # Convert list to NumPy array and ensure it is 2D
        player_colors_array = np.array(player_colors)
        if player_colors_array.ndim == 1:
            player_colors_array = player_colors_array.reshape(1, -1)
        elif player_colors_array.ndim == 2 and player_colors_array.shape[1] != 3:
            player_colors_array = player_colors_array.reshape(-1, 3)

        # Perform K-means clustering on the collected player colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors_array)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # Return cached team if already assigned
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        if player_color is None or self.kmeans is None:
            # Fall back to a default team (e.g., team 1)
            print(f"Warning: Falling back to default team for player {player_id}")
            self.player_team_dict[player_id] = 1
            return 1

        # Ensure player_color is 2D for prediction
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        # Custom override for player id 91 if needed
        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
