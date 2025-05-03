# Dataset 1 : Détection de Comptes Instagram Faux/Authentiques

## Source
Ce jeu de données provient de Kaggle : [Instagram Fake/Spammer/Genuine Accounts](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts)

## Description
Ce dataset contient des caractéristiques de comptes Instagram qui peuvent être utilisées pour identifier si un compte est authentique ou faux (fake). Il est conçu pour les tâches de classification binaire dans le domaine de la détection de fraude sur les réseaux sociaux.

## Caractéristiques (Features)
Le fichier `train.csv` contient les colonnes suivantes :

- `profile pic` : Présence d'une photo de profil (1 = Oui, 0 = Non)
- `nums/length username` : Ratio entre le nombre de chiffres et la longueur totale du nom d'utilisateur
- `fullname words` : Nombre de mots dans le nom complet
- `nums/length fullname` : Ratio entre le nombre de chiffres et la longueur totale du nom complet
- `name==username` : Indique si le nom et le nom d'utilisateur sont identiques (1 = Oui, 0 = Non)
- `description length` : Longueur de la description/bio du profil
- `external URL` : Présence d'une URL externe dans la bio (1 = Oui, 0 = Non)
- `private` : Compte privé ou public (1 = Privé, 0 = Public)
- `#posts` : Nombre de publications
- `#followers` : Nombre d'abonnés
- `#follows` : Nombre d'abonnements
- `fake` : Variable cible/étiquette (1 = Compte faux, 0 = Compte authentique)

## Applications possibles
- Détection de faux comptes sur Instagram
- Analyse de comportements frauduleux sur les réseaux sociaux
- Développement de systèmes de sécurité pour les plateformes sociales

## Format
Fichier CSV avec 12 colonnes