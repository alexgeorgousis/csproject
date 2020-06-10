from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import time

def save_frames_as_gif(frames, path='./animations/', filename='mountaincar_random.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

env = gym.make("MountainCarContinuous-v0")
num_eps = 1

obs = env.reset()
frames = []

for _ in range(num_eps):
    done = False
    obs = env.reset()

    # while not done:
    for _ in range(500):
        frames.append(env.render(mode="rgb_array"))
        # time.sleep(0.02)
            
        obs, _, done, _ = env.step(env.action_space.sample())

env.close()
save_frames_as_gif(frames)