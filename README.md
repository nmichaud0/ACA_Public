# ACA_Public

[DONE]

- Write first script
- Arranged data treatment to easy cleaned and sequences of words
- Random Forest – Boosted + Grid Searching / XGBoost
- Multilayer Perceptrons
- SVM
- Sentence transformers (BERT multilingual) vs. tokens dataframe // are quite good
- Try if dimensionality reduction could improve performances // not good at all!
- Hard vs. Soft Voting Classifiers // Soft really seems to be better!

[IN PROGRESS]

- Check multilinguality of my transformer model and how could I improve that // I'll keep this for later
- Grid Searching --> make a new script with a class // the beggining of the package :)

[TODO]

- NN : check RNN

[COMMENTS]

- "distiluse-base-multilingual-cased" seems to lack a bit of dimensionality: Need to do some deeper research on a at-least 768 dimensions multilingual model that is capable of transforming sentences...
- Grid Searching --> search parameters for each model, then soft voting clf // xgboost & NN grids ???
- Random Searching instead of Grid to gain time; nonetheless it seems to be way more efficient; maybe, having a strong-learners ensemble might be weaker than a waeak-learners ensemble
- The goal now is to check RNN and have a better embeddings transformer!
- Though, I'll check how SVC behaves with grid-searching on embeddings, might be interesting...
- BERT Transformer: https://metatext.io/models/bert-base-multilingual-cased !!!
- Try big sample sizes with hard voting classifier
- 