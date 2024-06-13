# Face Recognition System Using Siamese Networks

## Introduction

This project demonstrates the implementation of a face recognition system using Siamese networks. Siamese networks are widely used in face recognition tasks because they can effectively measure the similarity between two images, making them ideal for verification and identification purposes.

## Approach

1. **Foundational Learning**: Our project began with understanding the basics of deep learning and neural networks.
2. **Basic Implementation**: Using the knowledge of Gradient Descent Backpropagation Algorithm and other fundamentals, we implemented a digit classifier with a single-layer neural network using the basic NumPy module.
3. **Model Optimization**: We learned about model optimization techniques like batch normalization and dropout to counter problems like overfitting, subsequently implementing the digit classifier model using the PyTorch framework.
4. **CNNs and Architectures**: We delved into the workings of Convolutional Neural Networks (CNNs) and studied various CNN architectures.
5. **Custom Architecture**: With a good understanding of CNNs, after many trials, we built our own custom architecture on the MNIST/CIFAR-10 dataset, efficiently performing classification tasks.
6. **Face Recognition**: Finally, we learned the concepts of face recognition and proceeded with its implementation using Siamese networks.

## The Algorithm

### Siamese Networks

- **Structure**: A Siamese network consists of two identical subnetworks sharing the same architecture and parameters. These twin networks process two separate inputs in parallel.
- **Functionality**: Siamese networks are primarily used for comparing pairs of data points, such as images or texts, converting inputs into lower-dimensional representations known as "embeddings." These embeddings are vectors of numerical values capturing essential features of the input data.
- **Training**: The network is trained to ensure that embeddings of similar input pairs are close in the embedding space, while those of dissimilar pairs are far apart. A crucial aspect is that the twin networks share the same set of weights, ensuring consistent and compatible mappings.

![Siamese Architecture](https://datahacker.rs/one-shot-learning-with-siamese-neural-network/)

### Contrastive Loss

This is the loss function used for learning the similarity function.

![Loss formula](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*QE2ccvCNw6HOW85e.png)

- **Formula**: 
  \[
  L = (1 - Y) \frac{1}{2} (D_w)^2 + (Y) \frac{1}{2} \{ \max(0, m - D_w) \}^2
  \]
  where \(D_w\) is the Euclidean distance between the outputs of the sister networks, and \(Y\) is either 0 or 1. If the images are from the same class, \(Y\) is 0; otherwise, \(Y\) is 1.

## Implementation

- **Siamese Network**: The core of the project is the Siamese network model, consisting of convolutional and fully connected layers that learn to extract discriminative features from face images, defined in the `SiameseNetwork` class.
- **Contrastive Loss**: Training the Siamese network is guided by the contrastive loss, defined in the `ContrastiveLoss` class. This loss function encourages the network to minimize the distance between images of the same person while maximizing the distance between images of different people.
- **Custom Dataloader**: The project uses the Olivetti Faces dataset, loaded and preprocessed by the `OlivettiFaces` class, which prepares pairs of face images with labels (0 for the same person, 1 for different people) for training and evaluation.
- **Training Loop**: The training loop trains the Siamese network using the prepared dataset, optimizing the model with the Adam optimizer and recording the loss history.

## Technology Stack

- PyTorch
- NumPy
- Torchvision
- Scikit-learn
- Matplotlib

## Result

Our model achieved an accuracy of 95.5%.

---

This structured documentation outlines the methodology and technical aspects of implementing a face recognition system using Siamese networks. It highlights the progression from fundamental deep learning concepts to the final model, emphasizing the practical application of theoretical knowledge in building an effective face recognition system.
