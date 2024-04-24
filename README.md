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
1) [Data Preparation & Cleaning](#data)
2) [Exploratory Analysis](#analysis)
   - [Number Game](#number)
   - [Tokenization](#token)
   - [Word Cloud](#word)
   - [Sentiment Analysis](#sentiment)
3) [Machine Learning](#machine)
   - [Naive Bayes](#naive)
   - [Logistic Regression](#logistic)
   - [Support Vector Machine](#support)
   - [Random Forest Classifier](#random)
   - [BERT](#bert)
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
[Back to Content Page](#repository)  
  
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
> For sentiment analysis, we used the the module TextBlob for natural language processing tasks.
> This would help us in identifying sentiments - positive ➕, negative ➖, neutral; from a piece of text.
> <img width="826" alt="sentiment" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/8b56f829-7aff-4cda-917f-27f9429f9f1f">

## <a id = "machine">🤖 Machine Learning</a>
[Back to Content Page](#repository)  
  
Machine learning is a bracnh of artificial intelligence that focuses on developing algorithms and statistical models that allow us to learn from our data and make any predicitons or decisions without explicitly programming it. Machine learning can identify patterns across large datasets that is impossible for the human to do so efficiently. The machine's performance can also be improved over time as they are more exposed to more data or by fine-tuning certain parameters.
### <a id = "naive"> Naive Bayes</a>
It is a classificaiton algorithm that assumes all predictors are independent of one another.
#### 👍 Advantages
> 🟢 Easy to understand and implement  
> 🟢 Can be trained quickly and make fast predictions  
> 🟢 Can solve multi-class prediction problems  
#### 👎 Disadvantages
> 🔴 Lousy estimator  

#### 📇 Results
Statistics  
<img width="636" alt="image" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/d5ec185f-249b-4170-a710-d930fba6c6ed">  

Confusion Matrix  
<img width="694" alt="image" src="https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/4a905cf3-b203-49f4-999d-02784a22f3fe">  
