# SC1015_MiniProject_Team1
![Social Media Report](https://github.com/donkey-king-kong/SC1015_MiniProject_Team1/assets/119853913/5713929d-5eb2-46ba-8bec-097dcfee265b)

## 🧠 Team Contributors
| S/N | Team Members (FCS3) | Part |
| :-: | :- | :- |
| 1 | Babu Sankar Nithin Sankar | Data Preparation & Cleaning, Data Visualisation, Machine Learning Models |
| 2 | Singh Gunraj | Exploratory Data Analysis, Presentation, Script, Github Repository & Report |
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
Our dataset is taken from Kaggle: [Detecting bullying tweets 🤬 | PyTorch LSTM & BERT 🤖](https://www.kaggle.com/code/ludovicocuoghi/detecting-bullying-tweets-pytorch-lstm-bert)

## 🔎 Repository Overview
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

## <a id = "analysis">🧼 Exploratory Analysis</a>
