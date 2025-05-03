# Dataset 2 : Classification de Comptes Instagram (Réels/Bots/Arnaques)

## Source
Ce jeu de données provient de Kaggle : [Instagram Fake/Spammer/Genuine Accounts](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts)

## Description
Ce dataset contient des caractéristiques de comptes Instagram qui peuvent être utilisées pour identifier si un compte est réel, un bot, ou lié à une arnaque. Il est conçu pour les tâches de classification multi-classes dans le domaine de la détection de fraude et d'activités malveillantes sur les réseaux sociaux.

## Caractéristiques (Features)
Le fichier `LIMFADD.csv` contient les colonnes suivantes :

- `Followers` : Nombre d'abonnés du compte
- `Following` : Nombre d'abonnements du compte (comptes suivis)
- `Following/Followers` : Ratio entre abonnements et abonnés
- `Posts` : Nombre de publications
- `Posts/Followers` : Ratio entre publications et abonnés
- `Bio` : Présence d'une biographie (Yes = Oui, N = Non)
- `Profile Picture` : Présence d'une photo de profil (Yes = Oui, N = Non)
- `External Link` : Présence d'un lien externe dans la bio (Yes = Oui, N = Non)
- `Mutual Friends` : Nombre d'amis mutuels
- `Threads` : Utilisation de la fonctionnalité Threads (Yes = Oui, N = Non)
- `Labels` : Variable cible/étiquette (Real = Compte réel, Bot = Compte automatisé, Scam = Compte d'arnaque)

## Applications possibles
- Détection de comptes automatisés (bots) sur Instagram
- Identification de comptes liés à des arnaques
- Analyse des caractéristiques distinctives entre comptes légitimes et frauduleux
- Développement de systèmes de sécurité pour les plateformes sociales

## Format
Fichier CSV avec 11 colonnes