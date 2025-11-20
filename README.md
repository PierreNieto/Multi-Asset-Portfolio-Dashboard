MULTI-ASSET PORTFOLIO DASHBOARD  
Projet ESILV — Python, Git & Linux for Finance  
(Module complet : Single Asset + Multi-Asset)

Ce projet a été réalisé dans le cadre du cours Python, Git & Linux for Finance.
L’objectif est de développer une plateforme financière interactive capable de :

- récupérer des données financières en temps réel
- afficher des dashboards interactifs
- exécuter des stratégies quantitatives
- simuler un portefeuille multi-actifs
- produire des rapports journaliers automatisés
- fonctionner 24/7 sur une machine Linux distante

Le projet est développé en binôme avec deux modules distincts :
- Quant A — Single Asset Analysis
- Quant B — Multi-Asset Portfolio Analysis

Les deux modules sont intégrés dans une seule application Streamlit.

--------------------------------------------------------------------

1) SINGLE ASSET ANALYSIS MODULE (Quant A)
Dossier : /app/single_asset/

Fonctionnalités :
- Analyse d’un actif unique (AAPL, CAC40, EUR/USD…)
- Récupération dynamique des prix (API, yfinance, web scraping)
- Backtesting de deux stratégies minimum
- Calculs : Sharpe ratio, Max Drawdown, Volatilité, Performance cumulée
- Graphiques interactifs : prix de l’actif + stratégie
- Paramètres ajustables dans l’interface
- Bonus : modèle prédictif (ARIMA, Régression, ML)

--------------------------------------------------------------------

2) MULTI-ASSET PORTFOLIO MODULE (Quant B)
Dossier : /app/portfolio/

Fonctionnalités :
- Chargement multi-actifs (minimum 3 actifs) via yfinance ou API
- Calcul des rendements journaliers
- Construction de portefeuilles :
  * Equal-weight
  * Pondérations personnalisées
  * Rebalancement simple
- Mesures de performance :
  * Rendement cumulé
  * Volatilité annualisée
  * Sharpe ratio
  * Matrice de corrélation
- Visualisations interactives :
  * Prix de chaque actif
  * Valeur cumulée du portefeuille
  * Comparaison actifs vs portefeuille
  * Graphiques interactifs Plotly

--------------------------------------------------------------------

APPLICATION STREAMLIT

La plateforme finale comporte :
- Un menu latéral
- Une page Single Asset
- Une page Multi-Asset Portfolio
- Une mise à jour automatique des données toutes les 5 minutes
- Une gestion robuste des erreurs API

Exécution locale :
streamlit run app/main.py

--------------------------------------------------------------------

DEPLOIEMENT LINUX & CRON

Le projet doit être déployé sur une machine Linux avec :
- Application Streamlit active 24/7
- Mise à jour automatique des données
- Rapport quotidien (généré via cron à 20h00)
- Rapports stockés dans /reports/
- Script cron inclus dans le dépôt

--------------------------------------------------------------------

STRUCTURE DU DÉPÔT

Multi-Asset-Portfolio-Dashboard/
│
├── app/
│   ├── main.py
│   ├── single_asset/
│   └── portfolio/
│         ├── data_loader.py
│         ├── portfolio_calc.py
│         ├── strategies.py
│         ├── plots.py
│         ├── page_portfolio.py
│
├── cron/
├── reports/
├── requirements.txt
└── README.txt

--------------------------------------------------------------------

TECHNOLOGIES UTILISÉES

- Python 3.9
- Streamlit
- Pandas / NumPy
- Plotly
- SciPy
- yfinance
- Git / GitHub
- Linux (VM, cron)

--------------------------------------------------------------------

WORKFLOW GIT (BINÔME)

Branches :
main
single_asset
portfolio

Procédure :
1. Chaque membre travaille sur sa branche dédiée
2. Commits propres et réguliers
3. Pull Request vers main
4. Review + merge
5. Intégration finale et déploiement

--------------------------------------------------------------------

AUTEURS

Lou-anne Peillon — Single Asset Module (Quant A)
Pierre Nieto — Multi-Asset Portfolio Module (Quant B)

--------------------------------------------------------------------

OBJECTIF FINAL

Une application financière interactive, robuste, professionnelle, déployée
sur Linux et fonctionnant 24/7, permettant d’analyser un actif unique ainsi
qu’un portefeuille multi-actifs complet.
