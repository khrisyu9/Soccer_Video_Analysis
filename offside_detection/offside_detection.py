import cv2
import numpy as np

def detect_offsides(frames, tracks, team_control, view_transformer, tolerance=0.0):
    """
    Annotate frames with offside calls based on team labels and perspective:
    - Draw offside line in yellow parallel to goal line (using inverse perspective).
    - Circle and label attackers offside when they receive the ball.

    Args:
      frames: list[BGR images]
      tracks: dict with 'players': list per frame of {track_id: {..., 'bbox', 'transformed_position', 'position', 'team', 'has_ball'}}
      team_control: array of team IDs controlling the ball each frame
      view_transformer: ViewTransformer instance with pixel->field and field->pixel mappings
      tolerance: float, small offset in field coords

    Returns:
      annotated_frames: list of BGR images
      offside_flags_list: list[list[int]]
    """
    # Precompute inverse perspective from field to pixel
    inv_matrix = cv2.getPerspectiveTransform(
        view_transformer.target_vertices,
        view_transformer.pixel_vertices
    )
    # Extract field width from target vertices
    field_width = view_transformer.target_vertices[0, 1]

    annotated_frames = []
    offside_flags_list = []

    for frame_num, frame in enumerate(frames):
        img = frame.copy()
        offence_team = None
        if team_control is not None and len(team_control) > frame_num:
            offence_team = int(team_control[frame_num])
        players = tracks.get('players', [])[frame_num]

        # Collect defender x in field coords
        defender_field_xs = []
        for pid, pdata in players.items():
            if pdata.get('team') is None or pdata['team'] == offence_team:
                continue
            pos = pdata.get('transformed_position') or pdata.get('position')
            if pos is not None:
                defender_field_xs.append(pos[0])
        # Debug
        print(f"Frame {frame_num}: Number of defenders detected = {len(defender_field_xs)}")

        offside_flags = []
        if len(defender_field_xs) >= 2:
            # Determine field threshold
            sorted_f = sorted(defender_field_xs)
            offside_field = sorted_f[-2] + tolerance
            # Define two field points on the offside line
            pts_field = np.array([
                [offside_field, 0],
                [offside_field, field_width]
            ], dtype=np.float32).reshape(-1, 1, 2)
            # Transform back to pixel space
            pts_pixel = cv2.perspectiveTransform(pts_field, inv_matrix)
            p1 = tuple(pts_pixel[0][0].astype(int))
            p2 = tuple(pts_pixel[1][0].astype(int))
            # Draw yellow line
            cv2.line(img, p1, p2, (0, 255, 255), 2)

            # Check attackers
            for pid, pdata in players.items():
                if pdata.get('team') != offence_team or not pdata.get('has_ball', False):
                    continue
                pos = pdata.get('transformed_position') or pdata.get('position')
                bbox = pdata.get('bbox')
                if pos is None or bbox is None:
                    continue
                if pos[0] > offside_field:
                    offside_flags.append(pid)
                    x1, y1, x2, y2 = map(int, bbox)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(img, center, 15, (0, 0, 255), 3)
                    cv2.putText(img, 'Offside', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Redraw line in red
                    cv2.line(img, p1, p2, (0, 0, 255), 2)

        annotated_frames.append(img)
        offside_flags_list.append(offside_flags)

    return annotated_frames, offside_flags_list
