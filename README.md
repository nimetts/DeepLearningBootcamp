# Fish Classification Using Artificial Neural Networks (ANN)

## Project Overview
This project aims to classify different species of fish using an artificial neural network (ANN). The dataset contains images of various fish species, which are processed and fed into a neural network for classification.

## Table of Contents
1. [Required Libraries](#required-libraries)
2. [Data Preparation](#data-preparation)
3. [Model Architecture](#model-architecture)
4. [Hyperparameter Optimization](#hyperparameter-optimization)
5. [Model Evaluation](#model-evaluation)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [Kaggle Notebook](#kaggle-notebook)

### Required Libraries
The following libraries are used in this project:
- **TensorFlow and Keras**: For building and training the neural network model.
- **NumPy and Pandas**: For data manipulation and analysis.
- **Matplotlib and Seaborn**: For data visualization.
- **scikit-learn**: For evaluation metrics like confusion matrix and classification report.
- **Keras Tuner**: For hyperparameter tuning.

### Data Preparation
1. **Dataset Structure**: The dataset consists of images categorized into different folders based on fish species.
2. **Image Preprocessing**: Images are resized and normalized for input into the model. A portion of the data is set aside for validation.
3. **Data Visualization**: The distribution of images across different fish species is visualized using bar plots.

### Model Architecture
- A sequential model is built using Keras, with:
  - Input layer.
  - Several dense layers with batch normalization and dropout for regularization.
  - An output layer with a softmax activation function for multi-class classification.

### Hyperparameter Optimization
To enhance the model's performance, hyperparameter optimization was performed using **Keras Tuner**. The following hyperparameters were tuned:

- **Number of Neurons**: The number of neurons in each dense layer was varied to determine the optimal architecture for the model.
- **Learning Rate**: Different learning rates were tested to find the best rate for convergence during training.
- **Batch Size**: Various batch sizes were evaluated to optimize training efficiency and model performance.
- **Dropout Rate**: Different dropout rates were implemented to reduce overfitting and improve generalization.

The **Random Search** method was employed to sample from the hyperparameter space. This approach efficiently explored combinations of parameters, and the model's performance was evaluated on a validation set. The best-performing model configuration was then selected for final training.

### Model Evaluation
- The best model is evaluated using validation data, with metrics such as accuracy and loss computed.
- A confusion matrix and classification report are generated to assess the model's performance across different fish species.

### Results
The final model achieved a validation accuracy of approximately **76.03%**. While the model showed decent performance, the classification report indicated that there were challenges in distinguishing between certain species, leading to lower precision and recall scores for several classes. The confusion matrix revealed specific classes where the model struggled, suggesting areas for improvement in data representation or model architecture.

### Visualizations
- The training and validation loss and accuracy are plotted to visualize the model's performance over epochs.

### Kaggle Notebook
For the complete code and detailed explanations, please refer to the Kaggle notebook: [Fish Dataset ANN](https://www.kaggle.com/code/nimetseyrek/fishdatasetann).
