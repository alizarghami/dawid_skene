import numpy as np

class Observation(object):
    def __init__(self, label):
        self.label = label


class DataGenerator(object):
    def __init__(self, class_size):
        self.class_size = class_size

    def generate(self):
        label = np.random.choice(self.class_size)
        return Observation(label)


class Channel(object):
    def __init__(self):
        pass

    def train(self, class_size):
        raise NotImplementedError("Should be implemented in sub class")

    def estimate(self, sample):
        raise NotImplementedError("Should be implemented in sub class")


class RandomConfusionMatrixChannel(Channel): 
    def __init__(self):
         super().__init__()
         
         self.class_size = None
         self.confusionMatrix = None

    def train(self, class_size, correct_prediction_rate):
        self.class_size = class_size
        if class_size < 2:
            raise ValueError("Matrix size must be at least 2x2.")
        
        # Initialize the matrix with the diagonal elements
        matrix = np.ones((class_size, class_size)) * correct_prediction_rate
        
        # Adjust off-diagonal elements to ensure positive values and column sums to 1
        for j in range(class_size):
            # Calculate the remaining sum for off-diagonal elements
            remaining_sum = 1 - correct_prediction_rate
            num_off_diagonal_elements = class_size - 1
            
            if num_off_diagonal_elements > 0:
                # Create random positive values summing up to remaining_sum
                off_diagonal_values = np.random.rand(num_off_diagonal_elements)
                off_diagonal_values /= off_diagonal_values.sum()  # Normalize to sum to 1
                off_diagonal_values *= remaining_sum  # Scale to desired sum
                
                # Place values in the matrix
                k = 0
                for i in range(class_size):
                    if i != j:
                        matrix[j, i] = off_diagonal_values[k]
                        k += 1
        
        self.confusionMatrix = matrix

    def estimate(self, sample):
        return np.random.choice(range(self.class_size), p=self.confusionMatrix[sample.label])