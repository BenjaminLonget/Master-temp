import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

LARGE_DECEPTIVE_MAZE_NUMERIC = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

def plot_coordinates(coordinates):

        cmap = mcolors.ListedColormap(['white', 'gray'])
        #map_data = np.logical_xor(map_data, 1).astype(int)
        plt.imshow(LARGE_DECEPTIVE_MAZE_NUMERIC, cmap=cmap, origin='upper', extent=(-len(LARGE_DECEPTIVE_MAZE_NUMERIC[0])/2, len(LARGE_DECEPTIVE_MAZE_NUMERIC[0])/2, -len(LARGE_DECEPTIVE_MAZE_NUMERIC)/2, len(LARGE_DECEPTIVE_MAZE_NUMERIC)/2))

        # for i, trajectory in enumerate(trajectories):
        #     x, y = zip(*trajectory)
        plt.plot(0, 2, color="green", marker='o', markersize=10, label="Start")
        plt.plot(0, -4, color="orange", marker='o', markersize=15, label="Goal")
        for point in coordinates:
            x, y = point
            plt.plot(x, y, color='red', marker='o', markersize=2)
        # for point in self.successfull_trajectories:
        #     x, y = point
        #     plt.plot(x, y, color='blue', marker='o', markersize=2)

        plt.legend()
        plt.title('Final coordinates during exploration of double-deceptive Map')
        plt.show()

if __name__ == '__main__':
    coords = pd.read_csv("tests/Deceptive_maze/LSTM_AE/Deceptive_maze_LSTM_simple_2/models/policy_1/final_coordinates_with_LSTM.csv")
    coords = coords.values.tolist()
    print(f"No of coordinates: {len(coords)}")
    plot_coordinates(coords)

