from tqdm import tqdm
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from offside_detection import detect_offsides

def main():
    pbar = tqdm(total=15, desc="Video Processing Progress", ncols=100)

    # Step 1: Read Video
    pbar.set_description("Step 1: Reading Video")
    video_frames = read_video('input_videos/Vinicius_Goal_2024_UCL_Final.mp4')
    pbar.update(1)

    # Step 2: Initialize tracker and get object tracks
    pbar.set_description("Step 2: Initializing Tracker and Getting Object Tracks")
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
    pbar.update(1)

    # Step 3: Add object positions to tracks
    pbar.set_description("Step 3: Adding Object Positions")
    tracker.add_position_to_tracks(tracks)
    pbar.update(1)

    # Step 4: Estimate camera movement
    pbar.set_description("Step 4: Estimating Camera Movement")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=False
    )
    pbar.update(1)

    # Step 5: Adjust positions based on camera movement
    pbar.set_description("Step 5: Adjusting Tracks for Camera Movement")
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    pbar.update(1)

    # Step 6: Apply view transformation to tracks
    pbar.set_description("Step 6: Applying View Transformation")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    pbar.update(1)

    # Step 7: Interpolate ball positions
    pbar.set_description("Step 7: Interpolating Ball Positions")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    pbar.update(1)

    # Step 8: Estimate speed and distance for tracks
    pbar.set_description("Step 8: Estimating Speed and Distance")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    pbar.update(1)

    # Step 9: Assign player teams
    pbar.set_description("Step 9: Assigning Player Teams")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in tqdm(enumerate(tracks['players']), total=len(tracks['players']), desc="Team Assignment per Frame", leave=False):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            if team is None:
                team = 1
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))
    pbar.update(1)

    # Step 10: Assign ball acquisition to players
    pbar.set_description("Step 10: Assigning Ball Acquisition")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in tqdm(enumerate(tracks['players']), total=len(tracks['players']), desc="Ball Assignment per Frame", leave=False):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    team_ball_control = np.array(team_ball_control)
    pbar.update(1)

    # Step 11: Draw object tracks on video frames
    pbar.set_description("Step 11: Drawing Object Tracks")
    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        team_ball_control,
        team_colors=team_assigner.team_colors
    )
    pbar.update(1)

    # Step 12: Draw camera movement on video frames
    pbar.set_description("Step 12: Drawing Camera Movement")
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    pbar.update(1)

    # Step 13: Draw speed and distance annotations
    pbar.set_description("Step 13: Drawing Speed and Distance")
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    pbar.update(1)

    # Step 14: Detect offside
    pbar.set_description("Step 14: Detecting Offside")
    output_video_frames, offside_flags = detect_offsides(
        output_video_frames,
        tracks,
        team_ball_control,
        view_transformer
    )
    pbar.update(1)

    # Step 15: Save the output video
    pbar.set_description("Step 15: Saving Video")
    save_video(output_video_frames, 'output_videos/output_video3.avi')
    pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    main()
