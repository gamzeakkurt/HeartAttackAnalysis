"Heart Attack Analysis with Neural Network" 

### Data Information

Description of data attributes:

1. Age : Age in years
2. Sex: Sex (0 : Female, 1: Male)
3. Cp: Chest pain (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
4. Trtbps: Resting blood pressure
5. Chol: Serum cholestoral in mg/dl
6. Bs:  Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. Restecg: Resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)
8. Thalachh: Maximum heart rate achieved
9. Exng: Exercise induced angina (1 = yes; 0 = no)
10. Oldpeak: ST depression induced by exercise relative to rest
11. Slp: The slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
12. Caa: The number of major vessels (0-3) colored by flourosopy
13. Thall : 3 = normal; 6 = fixed defect; 7 = reversable defect
14. Output : The predicted attribute - diagnosis of heart disease (0= less chance of heart attack 1= more chance of heart attack)

### Visualization of Data
Each attribute visualizes in below.

<p align="center"><img src="https://user-images.githubusercontent.com/37912287/114738572-c3a13a80-9d50-11eb-82fd-97a84a4da953.png" /></p>
<p align="center">
  <b>Figure 1.1</b>
</p>
 
 Heat map for attributes
  
  <p align="center"><img src="https://user-images.githubusercontent.com/37912287/114754688-842f1a00-9d61-11eb-832e-f449c4b1f3f8.png" /></p>
<p align="center">
  <b>Figure 1.2</b>
</p>
 
 We have seen clearly that chest pain type are a positive relationship with heart rate. 
 
 Output labels are unbalanced and '1' labels have the highest samples than '0' labels. In **Figure 1.3**, it can be seen it clearly.
  


<p float="middle">
  <img src="https://user-images.githubusercontent.com/37912287/114756048-ffdd9680-9d62-11eb-87c1-923154929742.png" width="400" />
  <img src="https://user-images.githubusercontent.com/37912287/114756255-39160680-9d63-11eb-82f0-a64b68c7d550.png" width="400" />
  <p align="center">
  <b>Figure 1.3</b>
</p>
</p>

### Preprocessing

Labels are unbalanced so we use the over- resampling method to solve this problem. Thus, each label has the same sample size. You can see it in **Figure 1.4**.

<p align="center"><img src="https://user-images.githubusercontent.com/37912287/114788029-71c8d680-9d89-11eb-83a4-6abf73689f78.png" /></p>
<p align="center">
  <b>Figure 1.4</b>
</p>
After resampling method, we used the z-normalization technique in order to scale the values. Thus, the variance is one and the standard deviation is 0.

### Cross Validation 

We used the cross-validation technique to test the effectiveness of machine learning models. We split the data into the proportion of 80% train and 20% test. 

### Neural Network Model

We used a sequential neural network model. We have different fully connected layers that are the size of **32**,**64**,**128**,**128**, and **1** respectively. We compiled the model using adam optimizer, loss function and accuracy metric. We benefited from the early-stopping function to stop training when accuracy is not improved 20 consecutive times. We reduced the learning rate when a metric has stopped improving 5 consecutive times. For the fitting model, we used **28** batch size and **200** epochs. Also, the model evaluated the test data. The accuracy is **93.93** in **Figure 1.5**.

<p align="center"><img src="https://user-images.githubusercontent.com/37912287/114791379-21ed0e00-9d8f-11eb-870f-64f72cdf0b46.PNG" /></p>
<p align="center">
  <b>Figure 1.5</b>
</p>

The confusion matrix which has this accuracy is in **Figure 1.6**.

<p align="center"><img src="https://user-images.githubusercontent.com/37912287/114791469-477a1780-9d8f-11eb-8421-127c30e2d1b5.PNG" /></p>
<p align="center">
  <b>Figure 1.6</b>
</p>
Lastly, performance metrics are in the below table. 

<p align="center"> <b> Table 1.0 </b></p>

<table>
    <thead>
         <tr>
            <th>Accuracy</th>
            <th>F1 Score</th>
            <th>Precision</th>
            <th>Recall</th>
         </tr>
    </thead>
    <tbody>
         <tr>
            <th>93.93</td>
            <th>58.10</th>
            <th>78.94</th>
            <th>63.64</th>
         </tr>
    </tbody>
</table>
