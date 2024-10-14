#objective1 : selection of algorithm -> isolation forest

import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# Isolation Tree class
class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.size = None

    def fit(self, X, current_height):
        # Stop when the max height is reached or there is only one data point
        if current_height >= self.height_limit or len(X) <= 1:
            self.size = len(X)
            return self
        
        # Randomly choose a feature and split value
        self.split_feature = random.randint(0, X.shape[1] - 1)
        min_value, max_value = X[:, self.split_feature].min(), X[:, self.split_feature].max()
        if min_value == max_value:  # No possible split
            self.size = len(X)
            return self
        
        self.split_value = random.uniform(min_value, max_value)
        
        # Split the data into two branches
        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask
        
        self.left = IsolationTree(self.height_limit)
        self.right = IsolationTree(self.height_limit)
        
        self.left.fit(X[left_mask], current_height + 1)
        self.right.fit(X[right_mask], current_height + 1)
        
        return self

    def path_length(self, X, current_height=0):
        if self.left is None and self.right is None:
            return current_height + c(self.size)  # External node (leaf)
        if X[self.split_feature] < self.split_value:
            return self.left.path_length(X, current_height + 1)
        else:
            return self.right.path_length(X, current_height + 1)


# Function for average path length for isolation trees
def c(n):
    if n <= 1:
        return 0
    return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n  # Harmonic number approximation


# Isolation Forest class
class IsolationForest:
    def __init__(self, n_trees=100, subsample_size=256):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.trees = []

    def fit(self, X):
        self.trees = []
        height_limit = math.ceil(math.log2(self.subsample_size))  # Maximum tree height
        
        for _ in range(self.n_trees):
            # Sample a random subset of data
            X_sample = X[np.random.choice(X.shape[0], self.subsample_size, replace=False)]
            tree = IsolationTree(height_limit)
            tree.fit(X_sample, current_height=0)
            self.trees.append(tree)
    
    def anomaly_score(self, X):
        # Average path length for each point
        scores = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            path_lengths = np.array([tree.path_length(X[i]) for tree in self.trees])
            # Normalize by average path length in a random binary search tree
            scores[i] = 2 ** (-np.mean(path_lengths) / c(self.subsample_size))
        
        return scores
    
    def predict(self, X, threshold=0.5):
        # Returns 1 for anomalies, -1 for normal points
        scores = self.anomaly_score(X)
        return np.where(scores >= threshold, 1, -1)
    
    def fit(self, X):
        self.trees = []
        height_limit = math.ceil(math.log2(self.subsample_size))  # Maximum tree height
    
        for _ in range(self.n_trees):
        # Ensure we don't sample more points than available
            n_samples = min(X.shape[0], self.subsample_size)
            
            # Sample a random subset of data
            X_sample = X[np.random.choice(X.shape[0], n_samples, replace=False)]
            tree = IsolationTree(height_limit)
            tree.fit(X_sample, current_height=0)
            self.trees.append(tree)


# Function to simulate a real-time data stream
def real_time_data_stream(stream_length=1000, frequency=0.1, noise_level=0.1, anomaly_chance=0.01):
    """
    Generates a real-time data stream with regular patterns, noise, and occasional anomalies.
    
    Args:
        stream_length (int): Total number of data points to generate in the stream.
        frequency (float): Frequency of the sine wave (regular pattern).
        noise_level (float): Amplitude of random noise.
        anomaly_chance (float): Probability of an anomaly occurring at each data point.
    
    Yields:
        float: A new data point in the stream.
    """

    for i in range(stream_length):
        # Generate regular pattern (sine wave) with noise
        regular_value = np.sin(2 * np.pi * frequency * i)  # Sine wave
        noise = np.random.normal(0, noise_level)  # Random noise
        
        # Occasionally introduce anomalies
        if random.random() < anomaly_chance:
            anomaly = random.uniform(5, 10)  # A spike or outlier
            yield regular_value + noise + anomaly
        else:
            yield regular_value + noise
        
        # Simulate real-time delay (optional)
        time.sleep(0.01)  # 10 ms delay to simulate streaming


# Function to visualize the real-time data stream and mark anomalies
def visualize_real_time_with_anomalies(stream_length=100, frequency=0.1, noise_level=0.1, anomaly_chance=0.01, n_trees=100, subsample_size=256, threshold=0.6):
    """
    Visualizes a real-time data stream, detects anomalies using Isolation Forest, and marks them on the graph.
    
    Args:
        stream_length (int): Total number of data points to generate in the stream.
        frequency (float): Frequency of the sine wave (regular pattern).
        noise_level (float): Amplitude of random noise.
        anomaly_chance (float): Probability of an anomaly occurring at each data point.
        n_trees (int): Number of trees in the Isolation Forest.
        subsample_size (int): Number of samples per tree in the Isolation Forest.
        threshold (float): Threshold for determining anomalies.
    """
    
    # Initialize Isolation Forest
    isolation_forest = IsolationForest(n_trees=n_trees, subsample_size=subsample_size)
    
    # Collect initial points to fit Isolation Forest
    initial_data = np.array([x for _, x in zip(range(subsample_size), real_time_data_stream(stream_length, frequency, noise_level, anomaly_chance))])
    
    # Fit Isolation Forest on the initial data
    isolation_forest.fit(initial_data.reshape(-1, 1))  # Reshape for a single feature
    
    # Prepare for real-time plotting
    plt.ion()  # Interactive mode on for real-time updates
    fig, ax = plt.subplots()
    data_points = []
    anomaly_points = []
    anomaly_indices = []
    
    for i, data_point in enumerate(real_time_data_stream(stream_length, frequency, noise_level, anomaly_chance)):
        data_points.append(data_point)
        
        # Get prediction for the current data point
        current_point = np.array([[data_point]])
        prediction = isolation_forest.predict(current_point, threshold=threshold)
        
        # Mark anomaly
        if prediction == 1:
            anomaly_points.append(data_point)
            anomaly_indices.append(i)
        
        # Clear the plot and re-plot the data
        ax.clear()
        ax.plot(data_points, label='Data Stream', color='blue')
        
        # Plot anomalies in red
        ax.scatter(anomaly_indices, anomaly_points, color='red', label='Anomalies', marker='x')
        
        # Add labels and legend
        ax.set_title('Real-time Data Stream with Anomalies')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        
        # Update the plot
        plt.pause(0.01)  # Pause to update the plot in real-time
    
    plt.ioff()  # Turn off interactive mode after the stream ends
    plt.show()

# Run the visualization with anomaly detection
if __name__ == "__main__":
    visualize_real_time_with_anomalies(stream_length=200, frequency=0.1, noise_level=0.2, anomaly_chance=0.05, n_trees=100, subsample_size=256, threshold=0.6)


