### 2.  Neural Network Training:
#### Model structure:
The first part is a feature encoder using DenseNet-121 (saved under `mtdp`) as the backbone, and the model parameters pre-trained by the histopathology images are used as model initialization. In the second part, we utilized a multi-layer perception (saved under `Model`), which consists of two linear transformation layers and a rectified linear unit (ReLU) non-linear activation layer, to reduce the dimensionality of feature map to the 5 desired classifications. 

#### Training process:
To reduce the time-cost of the training process, we extracted the features from the last convolutional block before the fully connected layer (FCN), which outputs a 1024-dimensional vector for each patch and only trained the multi-layer perception. Through multiple hyperparameter tuning, the best model performance is generated by a stochastic gradient descent (SGD) optimizer with a learning rate of 0.0001 updating the model parameters for each training cycle, or epoch, after which the Balanced Cross Entropy Loss was calculated. The loss and accuracy were continuously monitored in the training process for 100 epochs. To prevent overfitting problem, we utilized augmentation methods including vertical flipping, color transformation, and affine transformation. The augmentation methods were only applied on the training data, and the normalization method was applied to all datasets. Besides, we added a dropout layer after the rectified linear unit (ReLU) non-linear activation layer. The training and validation dataset was spilt at the WSI level with the ratio of 8:2. The final model was selected by the best accuracy for the validation dataset. 