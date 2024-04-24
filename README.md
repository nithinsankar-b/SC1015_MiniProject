# SC1015_MiniProject_Team1
![Social Media Report](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/5713929d-5eb2-46ba-8bec-097dcfee265b)

## 🧠 Team Contributors
| S/N | Team Members (FCS3) | Part |
| :-: | :- | :- |
| 1 | Babu Sankar Nithin Sankar | Data Preparation & Cleaning, Data Visualisation, Machine Learning Models |
| 2 | Singh Gunraj | Exploratory Data Analysis, Presentation, Script |
| 3 | Lau Zhan You | Data Preparation & Cleaning, Data Visualization, Presentation, Script, Github Repository & Report |

## ❓About / Problem Statement
Problem Statement  
In today's interconnected world, social media have become central to our communication and our means of self-expression. However, this also gives rise to cyberbullying which can affect individuals mental health and well-being. Our motivation is to cultivate a safer online environment.

Aim: Detect & classify offensive or abusive language in tweets as cyberbullying using machine learning algorithms

Objective:  
🟢 Build a cyberbullying tweet detection model capable of identifying cyberbullying tweets  
🟢 Classify tweets based on demographic attributes such as age, etc to enhance detection  
🟢 Gain insights on the demographics that is most affected by online harrassment  
🟢 Optimize our model by employing machine learning algorithms to train & fine-tune our model  

Possibilities that this insights can be beneficial for:  
🟢 Protecting individuals mental health and well-being on social media  
🟢 Aiding social media platorms in implementing and upholding their community standards with regards to cyberbullying  

## 📖 Datasets
Our dataset is taken from Kaggle: [Cyberbullying Classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)

## <a id ="repository">🔎 Repository Overview </a>
> This is a long documentation. Use this section links to quickly and conveniently jump to each section.  
1) [Data Preparation & Cleaning](#data)
2) [Exploratory Analysis](#analysis)
   - [Number Game](#number)
   - [Tokenization](#token)
   - [Word Cloud](#word)
   - [Sentiment Analysis](#sentiment)
3) [Machine Learning](#machine)
   - [Naive Bayes](#naive)
   - [Multinomial Logistic Regression](#logistic)
   - [Support Vector Machine](#support)
   - [Random Forest Classifier](#random)
   - [Bi-LSTM](#bert)
4) [Results + Comparison](#results)
5) [Conclusion](#conclusion)

## <a id = "data">🧼 Data Preparation & Cleaning</a>
### Raw Data
<img width="682" alt="image" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/ff102811-5fc5-42a6-a5a9-1d06635cd886">
  
### What we removed
> 🟢 Remove mentions (@username)  
> 🟢 Remove punctuations  
> 🟢 Remove URLs  
> 🟢 Remove extra whitespaces  
> 🟢 Remove stopwords  
> 🟢 Remove HTML characters (EG: "&amp")  
> 🟢 Remove numbers  
> 🟢 Remove picture links (EG: pic.twitter.com)  
> 🟢 Remove shortwords (Length <= 2)  
  
### Functions
<img width="730" alt="image" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/2d813fbc-7e2d-4749-b5fa-6f456b4084cc">


## Result
> 1st Column: Raw tweet  
> 2nd Column: Cyberbullying Category  
> 3rd Column: Cleaned Tweet  
<img width="862" alt="image" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/74fa15ba-a331-4186-9944-599e38032f16">

## <a id = "analysis">🔬 Exploratory Analysis</a> 
[Back to `Main` Content Page](#repository)  
  
To analyse and visualze the data we have cleaned to understand its underlying patterns, relationships and anomalies. We would be using data visualization techniques in hopes of generating insights that could help us better understand the data before applying any models or conducting any hypothesis testing.
### <a id = "number">🔢 Number Game</a>
> The "numbers game" is used in our exploratory data analysis where we systematically examined numerical data to identify patterns, trends and anmoalies.
> Here, we plot the number of tweets belonging to each category in the dataset as well as their relative percentages.  
> <img width="862" alt="numbersgame" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/6c8194f1-ad10-414d-94ab-67589bf57f15">

### <a id = "token">🪙 Tokenization</a>
> Here, we used tokenization to break down a piece of text like sentences or paragraphs into individual worlds or "tokens".
> From this plot, we can see the most common words in the tweets of our data.
> <img width="826" alt="Tokenization" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/af0bb271-ca2e-4438-8253-7d91fbcbe5ea">


### <a id = "word">🔠 Word Cloud</a>
> For this section, we used a WordCloud to present the most commonly seen words according to each **classified** category.
> The presence of each words in a tweet will increase its corresponding probability towards being classified into its respective category.

#### Word Cloud was generated for the following 
> 🟢 [Gender Categories](#gender)  
> 🟢 [Religion Categories](#religion)  
> 🟢 [Age Categories](#age)  
> 🟢 [Ethnicity Categories](#ethnicity)  
> 🟢 [Other Cyberbullying Categories](#other)  
> 🟢 [Not Cyberbullying Related](#noncyber)     

#### <a id = "gender"> 👫Gender Related </a>
>![gender](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/7edcdd10-cd1b-4d16-9c1c-01fb9fd026e2)

#### <a id = "religion"> 🙏Religion Related </a>
>![religion](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/f402a453-50b0-4df4-a5f9-b67a6d46a9c2)

#### <a id = "age"> 🧓👵Age Related </a>
>![age](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/2d560124-2535-46fb-9132-e073bd196a7d)

#### <a id = "ethnicity"> :accessibility: Ethnicity Related </a>
>![ethnicity](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/bef99f02-e7f0-4996-bcc9-1168336cfc30)

#### <a id = "other"> 💎 Ethnicity Related </a>
>![other](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/6c84016c-ade5-4a9a-8004-5d71b11f6f95)

#### <a id = "noncyber"> 🔮 Not Cyberbullying Related </a>
>![noncyber](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/3f44bd65-86c4-458b-a4a6-c7c1695cd573)

### <a id = "sentiment">📈 Sentiment Analysis</a>
> - For sentiment analysis, we used the the module TextBlob for natural language processing tasks. The sentiment analysis model considers various factors such as word polarity, intensity of sentiment, and context to determine the sentiment score for a given text.  
> - This would help us in identifying sentiments - positive (➕), negative (➖), neutral; from a piece of text.
>   
> <img width="826" alt="sentiment" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/8b56f829-7aff-4cda-917f-27f9429f9f1f">
  
The sentiment score represents the polarity of the text (Positive, Negative, Neutral). It is a floating point number ranging from -1.0 to +1.0.  
> - If the sentiment score is close to 1.0, it indicates a very positive sentiment.  
> - If the sentiment score is close to -1.0, it indicates a very negative sentiment.  
> - If the sentiment score is around 0.0, it indicates a neutral sentiment.  

## <a id = "machine">🤖 Machine Learning</a>
[Back to `Main` Content Page](#repository)  
  
Machine learning is a bracnh of artificial intelligence that focuses on developing algorithms and statistical models that allow us to learn from our data and make any predicitons or decisions without explicitly programming it. Machine learning can identify patterns across large datasets that is impossible for the human to do so efficiently. The machine's performance can also be improved over time as they are more exposed to more data or by fine-tuning certain parameters.

> - [Naive Bayes](#naive)  
> - [Multinomial Logistic Regression](#logistic)  
> - [Support Vector Machine](#support)  
> - [Random Forest Classifier](#random)  
> - [Bi-LSTM](#bert)  

### <a id = "naive"> 1️⃣ Naive Bayes</a>
[Back to Machine Learning Content Page](#machine)  
[Back to `Main` Content Page](#repository)  
  
- It is a classificaiton algorithm that assumes all predictors are independent of one another.  
- Naive Bayes Model is a simple yet powerful machine learning algorithm used for NLP applications like text classification tasks, particularly in natural language processing (NLP). It's based on Bayes' theorem with the "naive" assumption of feature independence. Despite its simplicity, Naive Bayes often performs well in practice. In our classification, it performs moderately accurate.
#### 👍 Advantages
> 🟢 Easy to understand and implement  
> 🟢 Can be trained quickly and make fast predictions  
> 🟢 Can solve multi-class prediction problems  
#### 👎 Disadvantages
> 🔴 Lousy estimator  

#### 📇 Results
Statistics  
> <img width="826" alt="image" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/d5ec185f-249b-4170-a710-d930fba6c6ed">  

Confusion Matrix  
> <img width="826" alt="image" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/4a905cf3-b203-49f4-999d-02784a22f3fe">  

ROC Curve
> - We included this ROC curve to illustrate the balance between true positive rate (TPR) and false positive rate (FPR) across different thresholds.
> - A model excels when its curve hugs the top-left corner, indicating high TPR and low FPR. Conversely, a curve closer to the diagonal line signifies poor ability to discriminate, no better than random chance.
>  
> ![roc curve](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/e8242cef-b4df-4ac7-95d0-651bd59c28b5)

Learning Curve
> A learning curve is a plot that shows how a model's performance, often measured by accuracy, changes as the size of the training dataset increases. It helps assess if the model benefits from more data and can reveal issues like overfitting or underfitting. Cross-validation scores are often included for a more reliable estimate of performance.
>
> **Difference between Learning Curve & ROC Curve**  
> Learning Curve:  
> ➡️ Shows how a model's performance changes with varying training dataset sizes.  
> ➡️ Plots training and validation (or test) error/accuracy against the size of the training dataset.  
> ➡️ Helps identify whether a model suffers from underfitting (high bias) or overfitting (high variance).
>   
> ROC Curve:  
> ➡️ Evaluates the performance of a binary classification model across different classification thresholds.  
> ➡️ Plots the true positive rate (TPR) against the false positive rate (FPR) for various threshold values.  
> ➡️ Provides insights into the trade-off between sensitivity (true positive rate) and specificity (true negative rate).  
> ➡️ The area under the ROC curve (AUC-ROC) summarizes the overall performance of the classifier.
>
> ![learning](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/12b200e4-ed66-48bc-be98-4ab8492ec835)

### <a id = "logistic"> 2️⃣ Multinomial Logistic Regression</a>
[Back to Machine Learning Content Page](#machine)  
[Back to `Main` Content Page](#repository)  
  
- Multinomial Logistic Regression extends Logistic Regression to handle multi-class classification tasks.  
- This is done by predicting probabilities for each class and selecting the class with the highest probability as the predicted output.
#### 👍 Advantages
> 🟢 Provides probabilities for each category, allowing for nuanced predictions and quantification of uncertainty.  
> 🟢 Enables decision-makers to assess the likelihood of different outcomes, aiding in informed decision-making.  
> 🟢 Facilitates understanding of how predictors influence category selection, enhancing model interpretability.
#### 👎 Disadvantages
> 🔴 Assumes independence of observations, which may not hold in all datasets.  
> 🔴 Violation can lead to biased parameter estimates and inaccurate inference.  
> 🔴 Typically needs a larger sample size compared to a binary logistic regression.  

#### 📇 Results
Statistics  
><img width="826" alt="statistics logistic" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/61bb4416-95e0-416f-a2e0-ccf8634ba52f">  

Confusion Matrix
>![confusion logistic](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/91165e45-c5bd-48dd-9c3a-311ff5194321)  

ROC Curve
>![roc logisti](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/edac2f89-b220-4069-9b52-41e264a7015e)  

Learning Curve
> ![learning logistic](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/e7eb1811-c267-45e6-94c8-b638f4a0dac4)

### <a id = "support"> 3️⃣ Support Vector Machine</a>
[Back to Machine Learning Content Page](#machine)  
[Back to `Main` Content Page](#repository)  
  
- SVM classification finds the best hyperplane to separate data into different classes, maximizing the margin between them.  
- It's effective for various classification tasks due to its ability to handle linear and non-linear separations through kernel functions.  
#### 👍 Advantages
> 🟢 Performs well even in high-dimensional spaces, making it suitable for complex datasets.  
> 🟢 Aims to maximize the margin between classes, leading to a more generalizable model and reducing the risk of overfitting.   
> 🟢 Can handle non-linear decision boundaries using kernel functions like polynomial, radial basis function (RBF), and sigmoid, providing flexibility in modeling complex relationships.  
#### 👎 Disadvantages
> 🔴 Training models can be computationally intensive, especially for large datasets.    
> 🔴 SVM is sensitive to noisy data and outliers, which can affect the placement of the decision boundary and degrade performance.    
> 🔴 The decision boundary produced by SVM may be difficult to interpret, especially in higher dimensions or with non-linear kernels, making it challenging to understand the underlying relationships in the data.  

#### 📇 Results
Statistics  
> <img width="826" alt="SVM Stats" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/310477c1-b2b7-484a-8db4-c345bca22ea6">  

Confusion Matrix
> ![svm confusion](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/f63de957-aafc-49af-8bca-3ec24fefef0a)  

ROC Curve
> ![svm roc](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/aa307f0d-0181-4b90-9cde-4a496fedea3f)  

Learning Curve
> ![svm learning curve](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/ff9ad1e4-20c3-4cda-b99f-673bf33768db)

### <a id = "random"> 4️⃣ Random Forrest Classifier</a>
[Back to Machine Learning Content Page](#machine)  
[Back to `Main` Content Page](#repository)  
    
- Random Forest Classifier is an ensemble learning technique for classification tasks.  
- It builds multiple decision trees and outputs the mode of the classes predicted by individual trees.  
- It's effective, versatile, and resistant to overfitting.tions through kernel functions.  
#### 👍 Advantages
> 🟢 Often produces highly accurate predictions, even without extensive hyperparameter tuning.   
> 🟢 By aggregating predictions from multiple decision trees, Random Forest is less prone to overfitting compared to individual decision trees.   
> 🟢 Random Forest can efficiently handle large datasets with many features and instances, making it suitable for complex problems.  
> 🟢 Can handle missing values in the dataset without the need for imputation, reducing preprocessing requirements.  
#### 👎 Disadvantages
> 🔴 The ensemble nature of it makes it less interpretable compared to simpler models, as it's challenging to trace predictions back to individual trees.     
> 🔴 Training this model can be computationally expensive, especially for large datasets with numerous trees and deep trees.      
> 🔴 Random Forest tends to be biased towards the majority class in imbalanced datasets, potentially leading to suboptimal performance for minority classes.   

#### 📇 Results
Statistics  
> <img width="826" alt="forest statistics" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/f0c9136e-647a-457a-b3fe-27e0dea45a3d">

Confusion Matrix
> ![forest confusion](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/6ee0caf5-7b30-4c1f-917f-c4153ef6b62c)

ROC Curve
> ![forest roc](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/1ca65944-bbba-46b4-8298-e4170a88693b)

Learning Curve
> ![forest learning curve](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/30b742f3-9670-4101-9833-c0c3ec5f3a0d)

### <a id = "bert"> 5️⃣ Bi-LSTM</a>
[Back to Machine Learning Content Page](#machine)  
[Back to `Main` Content Page](#repository)  
  
- Also known as Bidirectional Long Short-Term Memory.    
- A type of recurrent neural network (RNN) that consists of 2 LSTM layers - processing in forward and backward directions.  

#### 👍 Advantages
> 🟢 Proccesses input sequences in both forward and backward directions helps in understanding the complete context of the input sequence.   
> 🟢 Well-suited for capturing long-term dependencies in sequential data and can effectively model complex dependencies over extended sequences.   
> 🟢 Random Forest can efficiently handle large datasets with many features and instances, making it suitable for complex problems.  
> 🟢 The gated architecture of LSTM cells helps mitigate the vanishing gradient problem, making it more capable of learning and retaining information over long sequences.    
#### 👎 Disadvantages
> 🔴 Effectively doubles the computational cost of processing each input sequence compared to unidirectional LSTMs.     
> 🔴 Require more memory to store the activations and gradients for both forward and backward processing directions.      
> 🔴 Complex models with multiple layers and bidirectional processing, make them less interpretable compared to simpler models.  
> 🔴 Prone to overfitting, especially when trained on small datasets or when the model capacity is too high relative to the dataset size.  

#### 📇 Results
> Due to the lengthy code, please refer to the source code for more detailed analysis.  

Classification Report
><img width="826" alt="Bi-lstm" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/2ec54521-c2da-4b27-a572-38a075f586c7">
