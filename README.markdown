# Alzheimer’s Disease Prediction Project

## Overview

This project aims to predict Alzheimer’s disease diagnosis using a dataset from Kaggle containing various patient features. Multiple machine learning classifiers are applied, and their performance is evaluated to select the best model based on classification reports.

## Dataset

The dataset consists of 2149 patient records with 35 features, including demographic, medical, and cognitive variables. Below is a summary of the dataset:

- **Rows**: 2149 (non-null for all features)
- **Columns**: 35
- **Features**:
  - **PatientID**: Unique identifier (int64)
  - **Age**: Patient age (int64)
  - **Gender**: Gender (int64)
  - **Ethnicity**: Ethnicity (int64)
  - **EducationLevel**: Education level (int64)
  - **BMI**: Body Mass Index (float64)
  - **Smoking**: Smoking status (int64)
  - **AlcoholConsumption**: Alcohol consumption level (float64)
  - **PhysicalActivity**: Physical activity level (float64)
  - **DietQuality**: Diet quality score (float64)
  - **SleepQuality**: Sleep quality score (float64)
  - **FamilyHistoryAlzheimers**: Family history of Alzheimer’s (int64)
  - **CardiovascularDisease**: Presence of cardiovascular disease (int64)
  - **Diabetes**: Diabetes status (int64)
  - **Depression**: Depression status (int64)
  - **HeadInjury**: History of head injury (int64)
  - **Hypertension**: Hypertension status (int64)
  - **SystolicBP**: Systolic blood pressure (int64)
  - **DiastolicBP**: Diastolic blood pressure (int64)
  - **CholesterolTotal**: Total cholesterol level (float64)
  - **CholesterolLDL**: LDL cholesterol level (float64)
  - **CholesterolHDL**: HDL cholesterol level (float64)
  - **CholesterolTriglycerides**: Triglycerides level (float64)
  - **MMSE**: Mini-Mental State Examination score (float64)
  - **FunctionalAssessment**: Functional assessment score (float64)
  - **MemoryComplaints**: Presence of memory complaints (int64)
  - **BehavioralProblems**: Presence of behavioral problems (int64)
  - **ADL**: Activities of Daily Living score (float64)
  - **Confusion**: Presence of confusion (int64)
  - **Disorientation**: Presence of disorientation (int64)
  - **PersonalityChanges**: Presence of personality changes (int64)
  - **DifficultyCompletingTasks**: Difficulty completing tasks (int64)
  - **Forgetfulness**: Presence of forgetfulness (int64)
  - **Diagnosis**: Target variable (int64, Alzheimer’s diagnosis)
  - **DoctorInCharge**: Doctor in charge (object)

## Models

The following machine learning classifiers were implemented to predict Alzheimer’s diagnosis:

- **Random Forest**: `RandomForestClassifier(random_state=42)`
- **Logistic Regression**: `LogisticRegression(random_state=42)`
- **K-Nearest Neighbors**: `KNeighborsClassifier()`
- **Support Vector Machine**: `SVC(random_state=42)`

## Methodology

1. **Data Preprocessing**:
   - Handle any missing values (none in this dataset).
   - Encode categorical variables (e.g., `DoctorInCharge` if used).
   - Scale numerical features if required (e.g., for SVM and KNN).
2. **Model Training**:
   - Split the dataset into training and testing sets.
   - Train each classifier on the training set.
3. **Evaluation**:
   - Generate classification reports (precision, recall, F1-score, accuracy) for each model.
   - Compare models based on classification report metrics to select the best-performing model.

## Dependencies

- Python 3.x
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `numpy`

Install dependencies using:

```bash
pip install pandas scikit-learn numpy
```

## Usage

1. Clone the repository or download the dataset from Kaggle.
2. Place the dataset in the project directory.
3. Run the script to train models and evaluate performance:

```bash
python alzheimers_prediction.py
```

4. Review the classification reports in the output to identify the best model.

## Results

The best model is selected based on the highest performance metrics (e.g., accuracy, F1-score) from the classification reports. Detailed results can be found in the console output or saved reports.

## Future Work

- Explore feature selection to improve model performance.
- Test additional models (e.g., Gradient Boosting, Neural Networks).
- Perform hyperparameter tuning for better accuracy.
- Incorporate cross-validation for robust evaluation.

## License

This project is licensed under the MIT License.