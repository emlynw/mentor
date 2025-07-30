#!/usr/bin/env python3

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from sd_vae_decode import latent_to_rgb

# Path containing your .npz replay-episode files
DEMO_BUFFER_PATH = "/home/emlyn/rl_franka/mentor/exp_local/2025.07.30/152418_/buffer"

def main():
    # Collect all .npz files in DEMO_BUFFER_PATH
    npz_files = sorted(glob.glob(os.path.join(DEMO_BUFFER_PATH, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {DEMO_BUFFER_PATH}")
        return

    for ep_file in npz_files:
        print(f"Loading: {ep_file}")
        # Load the episode dictionary
        episode_data = np.load(ep_file)
        # Convert to python dict with arrays
        episode = {k: episode_data[k] for k in episode_data.keys()}

       
        T = episode['emb'].shape[0]
        print(f"Episode length: {T} transitions")

        for t in range(T):
            # wrist1_emb is first half of emb, wrist2_emb is second half
            wrist1_emb = episode['emb'][t][:len(episode['emb'][t])//2]
            wrist2_emb = episode['emb'][t][len(episode['emb'][t])//2:]

            img_wrist1 = latent_to_rgb(wrist1_emb)
            img_wrist2 = latent_to_rgb(wrist2_emb)


            # 2) Convert from RGB -> BGR for OpenCV
            if img_wrist1.ndim == 3 and img_wrist1.shape[-1] == 3:
                img_wrist1_bgr = cv2.cvtColor(img_wrist1, cv2.COLOR_RGB2BGR)
            else:
                # If you have a different shape, adapt here
                if img_wrist1.shape[-1] == 2:
                    zeros = np.zeros((*img_wrist1.shape[:2], 1), dtype=img_wrist1.dtype)
                    img_wrist1 = np.concatenate([img_wrist1, zeros], axis=-1)
                    img_wrist1_bgr = cv2.cvtColor(img_wrist1, cv2.COLOR_RGB2BGR)


            if img_wrist2.ndim == 3 and img_wrist2.shape[-1] == 3:
                img_wrist2_bgr = cv2.cvtColor(img_wrist2, cv2.COLOR_RGB2BGR)
            else:
                if img_wrist2.shape[-1] == 2:
                    zeros = np.zeros((*img_wrist2.shape[:2], 1), dtype=img_wrist2.dtype)
                    img_wrist2 = np.concatenate([img_wrist2, zeros], axis=-1)
                    img_wrist2_bgr = cv2.cvtColor(img_wrist2, cv2.COLOR_RGB2BGR)

            # 3) Side-by-side horizontal concatenation
            vis = cv2.hconcat([img_wrist1_bgr, img_wrist2_bgr])


            # 4) Overlay text (reward, action, state, etc.) as needed
            reward = episode['reward'][t] if 'reward' in episode else 0.0
            action = episode['action'][t] if 'action' in episode else None
            state = episode['state'][t] if 'state' in episode else None
            terminated = bool(episode['terminated'][t]) if 'terminated' in episode else False
            truncated = bool(episode['truncated'][t]) if 'truncated' in episode else False

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (0, 0, 255)  # red
            line_type = cv2.LINE_AA

            cv2.putText(vis, f"File: {os.path.basename(ep_file)}", (10, 30),
                        font, 1.0, text_color, 2, line_type)
            cv2.putText(vis, f"Step: {t}/{T-1}", (10, 70),
                        font, 0.7, text_color, 2, line_type)
            cv2.putText(vis, f"Reward: {float(reward):.5f}",
                        (10, 110), font, 0.7, text_color, 2, line_type)

            if action is not None:
                # Action might be multi-dimensional; just show a short version
                action_str = np.array2string(action, precision=2, suppress_small=True)
                cv2.putText(vis, f"Action: {action_str}",
                            (10, 150), font, 0.7, text_color, 2, line_type)

            if state is not None:
                # State might be multi-dimensional
                state_str = np.array2string(state, precision=2, suppress_small=True)
                cv2.putText(vis, f"State: {state_str}",
                            (10, 190), font, 0.7, text_color, 2, line_type)

            cv2.putText(vis, f"Terminated: {terminated}", (10, 230),
                        font, 0.7, text_color, 2, line_type)
            cv2.putText(vis, f"Truncated: {truncated}", (10, 270),
                        font, 0.7, text_color, 2, line_type)
            
            

            # 5) Show
            cv2.imshow("Replay Viewer", vis)
            key = cv2.waitKey(0)  # 0 => wait until key press
            if key == 27:  # ESC pressed => exit entirely
                print("User pressed ESC. Exiting.")
                cv2.destroyAllWindows()
                return
            # Any other key => proceed to next frame

        print(f"Finished viewing {ep_file}.\n")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
