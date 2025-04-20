import os
import gymnasium as gym
import imageio

def record_episode(agent, env_name, num_episodes: int = 1, fps: int = 30):
    """
    Runs `num_episodes` episodes of `agent` in `env_name`,  
    captures rgb_array frames, and saves them as MP4s to ./recordings/<env_name>/.
    """
    # 1) Ensure output directory exists
    video_folder = os.path.join("recordings", env_name)
    os.makedirs(video_folder, exist_ok=True)

    for ep in range(num_episodes):
        # 2) Make a fresh env with rgb_array render mode
        env = gym.make(env_name, render_mode="rgb_array")
        obs, _ = env.reset()
        frames = []
        done = False

        # 3) Run one episode, grabbing frames
        while not done:
            frame = env.render()              # returns an HxWx3 RGB array
            frames.append(frame)

            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.close()

        # 4) Write out the video
        video_path = os.path.join(video_folder, f"{env_name}_episode_{ep}.mp4")
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Saved episode {ep} to {video_path}")
