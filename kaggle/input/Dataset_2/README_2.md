# Dataset 2: Instagram Account Classification (Real/Bot/Scam)

## Source
This dataset comes from Kaggle: [Instagram Fake/Spammer/Genuine Accounts](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts)

## Description
This dataset contains features of Instagram accounts that can be used to identify whether an account is real, a bot, or linked to a scam. It is designed for multi-class classification tasks in the domain of fraud detection and malicious activities on social networks.

## Features
The `LIMFADD.csv` file contains the following columns:

- `Followers`: Number of account followers
- `Following`: Number of accounts followed
- `Following/Followers`: Ratio between following and followers
- `Posts`: Number of posts
- `Posts/Followers`: Ratio between posts and followers
- `Bio`: Presence of a biography (Yes, N = No)
- `Profile Picture`: Presence of a profile picture (Yes, N = No)
- `External Link`: Presence of an external link in the bio (Yes, N = No)
- `Mutual Friends`: Number of mutual friends
- `Threads`: Use of the Threads feature (Yes, N = No)
- `Labels`: Target variable/label (Real = Real account, Bot = Automated account, Scam = Scam account)

## Possible Applications
- Detection of automated accounts (bots) on Instagram
- Identification of accounts linked to scams
- Analysis of distinctive features between legitimate and fraudulent accounts
- Development of security systems for social platforms

## Format
CSV file with 11 columns