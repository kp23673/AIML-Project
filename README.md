# AIML-Project:
 Project Name : OLA - Driver Sustain  Ensemble

# Introduction (Problem Statement) :
 OLA is leading transportation industry. Reducing drivers is seen by industry observers as a tough
 battle for Ola. Churn among drivers is high and it’s very easy for drivers to stop working for the
 service on the fly or jump to other transportation services depending on the rates.

 # Problem Statement :
 As the companies get bigger, the high churn could become a bigger problem. To find new
 drivers, Ola is casting a wide net, including people who don’t have cars for jobs. But this
 acquisition is really costly. Losing drivers frequently impacts the morale of the organization and
 acquiring new drivers is more expensive than retaining existing ones.
 
 You are working as a Data Scientist with the Analytics Department of Ola, focused on driver team
 attrition. You are provided with the monthly information for a segment of drivers for 2019 and
 2020 and tasked to predict whether a driver will be leaving the company or not based on their
 attributes like:
 
 Demographics (city, age, gender etc.)
 Tenure information (joining date, Last Date)
 Historical data regarding the performance of the driver (Quarterly rating, Monthly business
 acquired, grade, Income)

 # Dataset:
ola_driver.csv

# Concepts Tested:

Ensemble Learning- Bagging
Ensemble Learning- Boosting
KNN Imputation of Missing Values
Working with an imbalanced dataset

 # Results :
 
 1. Before diving into specific recommendations, it's essential to understand what ensemble
 learning is and how it works.
 2. Ensemble learning combines multiple machine learning models
 to improve prediction accuracy and robustness over individual models.
 3. There are various ensemble methods such as bagging, boosting, and stacking.
 4. From the analysis and feature selections, we get the idea how much driver are working and
 leaving.
 5. Encoding used for data efficiency.
 6. Played with Imbalanced and Balanced data using Logistic Regression, KNN classifier, DecisionTree Classifier, RandomForest, Hyperparameter Tuning, Bagging Boosting to see which algorithm is good for this data.
 7. By doing Comparison between balanced and imbalanced data we get know that balanced isgood compared to imbalanced.
 8. By ROC curve we get know balanced data is about 90% which is better.
 9. Experiment with different types of algorithms to capture diverse patterns in the data

# Conclusion : 
 1. In conclusion, the analysis of driver team attrition at Ola presents several key findings and
 recommendations.
 2.Firstly, the high churn rate among drivers poses a significant challenge for the company, impacting morale and incurring substantial costs associated with driver acquisition.
 3. Through the examination of monthly data for 2019 and 2020, it is evident that demographic factors such as age, gender, and city, alongside tenure information and historical
 performance metrics, play crucial roles in predicting driver attrition.
 4. Leveraging ensemble learning techniques such as bagging and boosting, as well as KNN imputation for handling missing values, proves to be effective in developing predictive
 models for identifying drivers at risk of leaving the company.
 5. Additionally, given the imbalanced nature of the dataset, strategies for working with imbalanced data, such as oversampling or incorporating class weights, are essential for
 achieving accurate predictions.
 6. Moving forward, Ola can use these insights to implement targeted retention strategies, focusing on factors identified as significant predictors of attrition.
 7. By addressing the root causes of driver churn and prioritizing the retention of existing drivers, Ola can mitigate the adverse effects of high turnover rates and ensure the stability
 and sustainability of its driver workforce.

 
