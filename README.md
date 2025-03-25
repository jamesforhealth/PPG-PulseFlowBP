# Precisely Filtered Cuffless Blood Pressure Prediction using Multi-Modal Data from PulseDB

## Project Overview

This project focuses on the development and evaluation of various machine learning models for cuffless blood pressure (BP) prediction using a precisely filtered, high-quality subset of the PulseDB dataset. Our research explores different modeling strategies, including building a general model applicable across individuals, fine-tuning this general model for specific individuals, and training dedicated models for each individual. Importantly, our approach leverages a combination of features derived from Photoplethysmography (PPG) signals, and potentially Electrocardiography (ECG) signals and personal information, along with explicitly incorporating relevant cardiovascular-related features within our regression methods.

## Dataset

This project utilizes a meticulously curated and **precisely filtered** subset of the **PulseDB** dataset. A significant effort has been made to **remove noisy data and data contaminated by artifacts**, ensuring a high-quality training resource for cuffless blood pressure prediction research. This refined dataset contains PPG signals, and may also include synchronized ECG signals and relevant personal information for each subject.

## Methodology

We explore several approaches to predict blood pressure without the need for a traditional cuff, utilizing a blend of input features and regression techniques:

1.  **General Model:**
    * We aim to build a robust machine learning model trained on the entire filtered PulseDB dataset (or a significant portion thereof) to predict blood pressure for a general population.
    * This model utilizes a combination of features, primarily derived from **PPG signals, and potentially incorporating ECG signals and personal information**.
    * **Crucially, our regression methods also incorporate cardiovascular-related features** extracted from the physiological signals, going beyond raw signal data. These features are designed to capture key physiological indicators relevant to blood pressure.
    * We will experiment with various regression models, potentially including but not limited to:
        * Traditional machine learning models (e.g., Random Forest, Gradient Boosting).
        * Deep learning models (e.g., Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformer networks) designed for time-series data, with architectures adapted to handle multi-modal input and extracted features.

2.  **Personalized Fine-tuning:**
    * We will investigate the effectiveness of fine-tuning the pre-trained general model on the data of individual subjects.
    * This approach aims to adapt the general knowledge learned by the model to the specific physiological characteristics of each individual, potentially leading to improved prediction accuracy compared to the general model alone.

3.  **Personal Model:**
    * We will also explore training individual machine learning models for each subject in the filtered PulseDB dataset, using only their respective data.
    * This strategy allows the model to learn highly personalized relationships between the blended features (derived from PPG, potentially ECG, personal information, and cardiovascular insights) and blood pressure for each individual.

## Input Signals and Features

Our models leverage a combination of information for blood pressure prediction:

* **Photoplethysmography (PPG):** The primary input signal, from which various temporal and morphological features are extracted.
* **Electrocardiography (ECG):** May be used as an additional signal source to derive complementary features related to cardiac timing and function.
* **Personal Information:** Such as age, gender, weight, height, etc., which may have correlations with blood pressure and serve as valuable contextual features.
* **Cardiovascular-Related Features:** Explicitly engineered features derived from the PPG and potentially ECG signals, designed to capture physiological indicators known to be associated with blood pressure regulation.

## Expected Outcomes

This project aims to achieve the following:

* Develop a robust general machine learning model capable of predicting cuffless blood pressure with high accuracy on the precisely filtered PulseDB dataset, utilizing a blend of signal features and cardiovascular insights.
* Demonstrate the significant potential of personalized fine-tuning to enhance blood pressure prediction accuracy for individual subjects.
* Evaluate the performance of individual-specific models trained on single-subject data using our multi-feature approach.
* Compare the effectiveness of the general model, personalized fine-tuned models, and individual models in the context of cuffless blood pressure prediction.
* Contribute to the advancement of precise and reliable non-invasive blood pressure monitoring techniques.

## Tools and Technologies

* **Python:** Primary programming language.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical computations.
* **Scikit-learn:** For implementing traditional machine learning models and evaluation metrics.
* **TensorFlow or PyTorch:** For building and training deep learning models.
* **Libraries for signal processing:** (e.g., SciPy, librosa) for extracting features from PPG and ECG signals.
* **Libraries for feature engineering:** (e.g., custom functions developed for extracting cardiovascular-related features).
* **Matplotlib and Seaborn:** For data visualization and result presentation.

## Potential Future Work

Building upon the findings of this project, future research could explore:

* Investigating more sophisticated deep learning architectures tailored for multi-modal time-series data and feature fusion.
* Developing novel cardiovascular-related features to further improve prediction accuracy.
* Exploring the use of explainable AI (XAI) techniques to understand the model's predictions and the importance of different features.
* Investigating the robustness and generalizability of the models to unseen populations and different data acquisition settings.
* Exploring the integration of these models into real-time wearable blood pressure monitoring systems.

## Contact Information
James Lin - AI/ML Algorithm Researcher jameslin@flowehealth.com