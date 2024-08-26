
import matplotlib.pyplot as plt

# Plot our dimensionality-reduced (via PCA) dataset.
def plot_data(x_component,y_component,y_raw):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(8.5, 6), dpi=130)
        plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=5, alpha=8/10)
        plt.title('classes after PCA transformation')
        plt.show()
        
        
# Visualize the training data.

def plot_is_label(transformed_data,indices):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(6, 6), dpi=100)
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='0.8', label='unlabeled',s=5)
        plt.scatter(transformed_data[indices, 0], transformed_data[indices, 1], c='k', label='labeled',s=5)
        plt.title('Unlabeled and labeled data')
        plt.legend()
        plt.show()
        
        
# Plot our classification results.
def plot_is_correct(x_component,y_component,is_correct,score):
    with plt.style.context('seaborn-white'):
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
        ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct',s=5)
        ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect',s=5)
        ax.legend(loc='lower right')
        ax.set_title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=score))
        plt.show()
        
    