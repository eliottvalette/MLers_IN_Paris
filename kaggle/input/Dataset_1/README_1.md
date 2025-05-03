# Dataset 1: Instagram Fake/Genuine Account Detection

## Source
This dataset comes from Kaggle: [Instagram Fake/Spammer/Genuine Accounts](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts)

## Description
This dataset contains features of Instagram accounts that can be used to identify whether an account is genuine or fake. It is designed for binary classification tasks in the domain of social media fraud detection.

## Features
The `train.csv` file contains the following columns:

- `profile pic`: Presence of a profile picture (1 = Yes, 0 = No)
- `nums/length username`: Ratio between the number of digits and the total length of the username
- `fullname words`: Number of words in the full name
- `nums/length fullname`: Ratio between the number of digits and the total length of the full name
- `name==username`: Indicates whether the name and username are identical (1 = Yes, 0 = No)
- `description length`: Length of the profile description/bio
- `external URL`: Presence of an external URL in the bio (1 = Yes, 0 = No)
- `private`: Private or public account (1 = Private, 0 = Public)
- `#posts`: Number of posts
- `#followers`: Number of followers
- `#follows`: Number of accounts followed
- `fake`: Target variable/label (1 = Fake account, 0 = Genuine account)

## Possible Applications
- Detection of fake accounts on Instagram
- Analysis of fraudulent behavior on social networks
- Development of security systems for social platforms

## Format
CSV file with 12 columns