# Eye Disease Classification Using Dual-Branch CNN Architecture
<img width="200" alt="image" src="https://github.com/user-attachments/assets/1fb55b35-1dbf-4f55-9546-061b3d105570" />
<img width="194" alt="image" src="https://github.com/user-attachments/assets/48962e92-c50b-470b-8242-8b16ce959771" />
<img width="194" alt="image" src="https://github.com/user-attachments/assets/8a1b5dce-2088-434c-82fd-bee26eb886fb" />




## Project Overview
This project implements a dual-branch convolutional neural network architecture for the classification of eye diseases. The model combines a custom CNN with a pre-trained ResNet50 to achieve high accuracy in distinguishing between four eye conditions: cataracts, diabetic retinopathy, glaucoma, and normal eyes.

## Dataset
The dataset used in this project is "Eye Diseases Classification" from Kaggle, created by Guna Venkat Doddi. It contains 4,217 images across four classes:
- Cataract
- Diabetic Retinopathy
- Glaucoma
- Normal (healthy eyes)

The dataset was downloaded directly within the project using the Kaggle API:
```python
!kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification
!unzip eye-diseases-classification.zip
```

## Data Preprocessing and Augmentation
The dataset was split into:
- Training set (68% of total data)
- Validation set (12% of total data)
- Testing set (20% of total data)

Several augmentation techniques were applied to the training set to improve model generalization:
1. Random rotation (up to 20%)
2. Horizontal flipping
3. Random zooming (up to 10%)
4. Brightness adjustments (±0.5%)
5. Contrast adjustments (±0.5%)
6. Custom Gaussian noise addition (mean=0, std=0.05)

All images were resized to 256×256 pixels and rescaled to the [0,1] range.

## Model Architecture
The implemented model uses a dual-branch architecture:

### Branch 1: Custom CNN
- 4 convolutional blocks, each with:
  - Conv2D layer (filters: 32→64→128→256)
  - ReLU activation
  - MaxPooling2D
- Flattening layer
- Dense layer (64 units) with ReLU activation
- Dropout (50%)

### Branch 2: Pre-trained ResNet50
- ResNet50 base model (pre-trained on ImageNet)
- Global Average Pooling
- Dense layer (64 units) with ReLU activation

### Combined Architecture
- Concatenation of both branch outputs
- Dense layer (256 units) with ReLU activation
- Dropout (30%)
- Output layer (4 units) with softmax activation

## Training
- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Metric: Accuracy
- Batch size: 32

## Results
The model achieved impressive performance metrics:

### Overall Performance
- Accuracy: 84.71%
- Precision: 85.74%
- Recall: 84.71%
- F1-Score: 84.44%

### Class-Specific Performance
1. **Cataract**
   - Accuracy: 91.13%
   - Precision: 80.86%
   - Recall: 82.44%
   - F1-Score: 81.64%

2. **Diabetic Retinopathy**
   - Accuracy: 100.00%
   - Precision: 100.00%
   - Recall: 100.00%
   - F1-Score: 100.00%

3. **Glaucoma**
   - Accuracy: 88.80%
   - Precision: 88.24%
   - Recall: 63.38%
   - F1-Score: 73.77%

4. **Normal**
   - Accuracy: 89.50%
   - Precision: 73.06%
   - Recall: 92.09%
   - F1-Score: 81.48%

## Key Findings
- The model achieves perfect classification for diabetic retinopathy
- Cataract and normal eye classification perform well with high F1-scores
- Glaucoma detection has high precision but lower recall, indicating some false negatives

## Potential Improvements
1. **Address Class Imbalance**: Implement techniques like class weighting or SMOTE to improve performance on classes with lower recall.

2. **Fine-Tuning ResNet50**: Unfreeze some of the later layers of ResNet50 and fine-tune them specifically for eye disease detection.

3. **Ensemble Methods**: Implement bagging or boosting to combine multiple models for more robust predictions.

4. **Additional Augmentations**: Explore domain-specific augmentations that mimic real-world variations in eye imagery.

5. **Hyperparameter Optimization**: Use techniques like grid search or Bayesian optimization to find optimal hyperparameters.

6. **Attention Mechanisms**: Incorporate attention layers to help the model focus on the most disease-relevant parts of the eye images.

7. **Cross-Validation**: Implement k-fold cross-validation to get a more reliable estimate of model performance.

8. **Explainability Tools**: Add GradCAM or SHAP visualizations to improve model interpretation for clinical applications.

## Dependencies
- TensorFlow 2.x
- NumPy
- OpenCV
- SciPy
- Kaggle API (for dataset download)

## Usage
To train and evaluate the model, run the provided scripts sequentially:
1. Download and extract the dataset
2. Preprocess and augment the data
3. Train the dual-branch model
4. Evaluate performance metrics

## Conclusion
The dual-branch CNN architecture leveraging both a custom CNN and pre-trained ResNet50 proves effective for eye disease classification. The model's high performance across multiple metrics demonstrates its potential for clinical application, though further improvements could enhance its utility in real-world medical settings.
