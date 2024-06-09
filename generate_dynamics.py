from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
import json
import torch as th
import os

from agent import ENV_KWARGS
from inverse_dynamics_model import IDMAgent


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MESSAGE = """
This script will take a video, predict actions for its frames and
and show them with a cv2 window.

Press any button the window to proceed to the next frame.
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0


def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action

def process_video(agent, video_path, output_json_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    required_resolution = ENV_KWARGS["resolution"]
    json_data = []
    total_frames = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 360))

    batch_size = 128 
    frame_count = 0  
    while True:
        frames = []
        print("Loading frames...")  
        for _ in range(batch_size):  
            ret, frame = cap.read()
            if not ret:
                break
            frame_downsampled = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            out.write(frame_downsampled)  
            frames.append(frame[..., ::-1])  
            total_frames += 1

        if not frames:
            break  

        print(f"Processing batch: {frame_count + 1}") 
        frames = np.stack(frames)
        predicted_actions = agent.predict_actions(frames)
        for i in range(len(frames)):
            action_record = {action: predicted_actions[action][0, i].tolist() for action in predicted_actions}
            json_data.append(action_record)

        frame_count += len(frames)
        print(f"Processed {frame_count} frames so far...")  
        th.cuda.empty_cache()

    cap.release()
    out.release()

    if len(json_data) == total_frames:
        with open(output_json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print("All frames processed successfully and data saved.")
    else:
        print(f"Warning: Frame-action mismatch in {video_path} ({len(json_data)} actions for {total_frames} frames).")


def main(model, weights, directory_path, output_directory, mode, test_video_path=None):
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    if mode == 'test' and test_video_path:
        video_file = os.path.basename(test_video_path)
        output_json_path = os.path.join(output_directory, f"{video_file[:-4]}_actions.json")
        output_video_path = os.path.join(output_directory, f"{video_file[:-4]}_downsampled.mp4")
        process_video(agent, test_video_path, output_json_path, output_video_path)
    elif mode == 'launch':
        for folder_name in os.listdir(directory_path):
            folder_path = os.path.join(directory_path, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.mp4'):
                        video_path = os.path.join(folder_path, file_name)
                        output_json_path = os.path.join(output_directory, f"{file_name[:-4]}_actions.json")
                        output_video_path = os.path.join(output_directory, f"{file_name[:-4]}_downsampled.mp4")
                        process_video(agent, video_path, output_json_path, output_video_path)
    else:
        print("Invalid mode or test video path not provided for test mode.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Process videos to predict actions and downsample.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--directory-path", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['test', 'launch'], required=True, help="Choose 'test' to run a single video or 'launch' to process all videos.")
    parser.add_argument("--test-video-path", type=str, help="Path to the test video file (only needed for test mode).")

    args = parser.parse_args()
    main(args.model, args.weights, args.directory_path, args.output_directory, args.mode, args.test_video_path)