# Neural Network Based Handwriting Recognition Tool

## Objective
To enable a machine learning-driven handwriting recognition application that can convert the handwritten script into a computerized legible script, and continuously learn in the process.

## About the Dataset
Dataset used was created by the National Institute of Standards and Technology (NIST). The NIST Special Database 19 consists of roughly 0.7 million sample png images. The current model has been trained only for uppercase letters (A-Z).
Dataset Link: https://www.nist.gov/srd/nist-special-database-19

## Preprocessing
While each character in the original dataset occupies 128x128 pixels per raster (Fig 2a), to avoid heavy computation for this project we decided to first, reduce the size of the image to 56 x 56 pixels (Fig 2b), and then further reduce the canvas size to 28x28 pixels by removing the padding (Fig 2c) and thus creating a 784 feature configuration dataset. Each character is labeled sequentially from “A”- “Z”.
    (c)
→→
(a) (b)
Figure 2: (a) Original 128 x 128-pixel raster as obtained from the NIST Database, (b) 56 x 56- pixel raster upon resizing the image, (c) 28 x 28-pixel raster upon removing the padding from the resized image
The package ‘tkinter’ enables the programming of the user interface. The implementation of this package created a canvas platform where-in the user is able to enter their character or word inputs. The pop-up canvas will convert the image into a NumPy array, traveling column-wise through the canvas and looking for a filled pixel to mark the beginning of a letter. For words, the model continues to traverse and look for a column where there is significant relative blank space to mark the beginning of the second character. Through repeated training, we were able to train the model for precision, meaning the model is intelligent enough to differentiate a break in letters versus the beginning of a second letter. This will be pushed into the model to convert the handwritten scripted character, into a computerized font.
### Features
It has features such as Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

Features have been segmented into below mentioned High Level Categories to simplify GUI:    
* **Personal Features** --> Age, Education, EducationField, MaritalStatus, Gender. 
* **Organisation Features** --> DailyRate, Department, HourlyRate, JobInvolvement, JobLevel, JobRole, MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
* **Commutation Features** --> BusinessTravel, DistanceFromHome
* **Satisfaction Features** --> EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance

For better performance and as per the context, features have been labelled as:
* **Continuous Features** --> Age, DistanceFromHome, DailyRate, HourlyRate, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
* **Categorical Features** --> JobLevel, MaritalStatus, Department, OverTime, PerformanceRating, Gender, StockOptionLevel, WorkLifeBalance, BusinessTravel, JobSatisfaction, JobRole, RelationshipSatisfaction, Education, EducationField, EnvironmentSatisfaction, JobInvolvement

Also, non-continuous variables have also been tagged as:
* **Ordinal Variables** --> Education, JobLevel
* **Nominal Variables** --> BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime, StockOptionLevel


## Application

### Quick Notes on terms used

* Attrition: Yes --> Refers to subset of dataset with people whose attrition value is equal to yes (means who have exited the company)

* Attrition: No --> Refers to subset of dataset with people whose attrition value is equal to no (means who have remain in the company)

* Select Feature Category --> As per the categories mentioned above, user can select the feature category and then select the feature from the list under that category. This makes it easy to look for variables from a huge list.

* Select Feature --> As per the feature category selected, the list of features populates that relate to that category.

* Choose Data Filter --> User can select the filter on dataset to showcase the graph, eg: choosing the data filter to Attrition: Yes will subset the data only for values with Attrition equal to Yes. This has three options namely All Data, Attrition: Yes and Attrition: No

* Percentage for Test: Percentage of total data to be used for testing the performance of the model


### Overall application can be segmented into two major parts:

1. **EDA Analysis**: This is essentially used to evaluate features w.r.t entire dataset, Attrition Yes and Attrition No
    1. **Variable Distribution**: Showcases the distribution of continuous features along with option to choose data filter
    2. **Variable Relation**: Showcases the relationship between two features along with option to choose data filter. based on the type of features, the graph changes i.e., in case both the features are continuous, then a scatter plot is poulated, while in case one feature is categorical and the other is continuous, then a box plot is populated.
    3. **Attrition Relation**: Showcases the comparison of a feature between Attrition:Yes and Attrition:No subsets. For continuous features, it compares the Minimum, Median, Mean and Maximum Values. For categorical features, it compares the count of values for each distinct element of the feature.

2. **ML Models**: This is used to see how the features along with the type of model perform to predict the Attrition of an employee. User can:
    * Input the value for Test Percentage (set by default to 20)
    * Select the features from the list (by default all are selected)
    * View the performance of the model in terms of Accuracy, Precision, Recall and F1 Score
    * Compare the performance of the current model with the other available models with same configuration (features and test percentage)
    
    Following are the models used for classifying the Atrition of an employee as Yes or No:
    1. **Decision Trees**: Showcases the results in the form of Confusion Matrix, Decision Tree Graph, Importance of Features and ROC Curve by Class.
    2. **Random Forest**: Showcases the results in the form of Confusion Matrix, AUC Score vs Number of Trees, Importance of Features and ROC Curve by Class. User can input the number of trees (set by default to 35).
    3. **Logistic Regression**: Showcases the results in the form of Confusion Matrix, Calibration Curve, Cross Validation Score, and ROC Curve by Class.
    4. **K Nearest Neighbours**: Showcases the results in the form of Confusion Matrix, Accuracy vs K Value, Cross Validation Score, and ROC Curve by Class. User can input the number of neighbours (set by default to 9).
    





## Run the app
To start the tool, run the "code.py".  
(Mac Users: When the Application starts, unfocus the app and then focus the app again to activate the menu items).  
Note: In case if you wish to skip the installation of the required packages (such as PyQt5, sklearn etc), comment the lines 2-9.






ML Based Handwriting Recognition Tool

To run the GUI, kindly run the "final_code.py".

<strong>Note:</strong> To use the pre trained model in the GUI, "MLP_Adam_100_SelectedData_sigmoid_earlystop.sav", please ensure you have sklearn version 0.22.1 installed in Anaconda. Otherwise, kindly follow the below steps to run the tool:
1. Unzip the final_data_a_z.csv.zip.
2. Run the MLP_Classifier.py
3. Run the final_code.py to run the GUI.
