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
- Grid Searching --> make a new script with a class // the beggining of the package :)
- Check multilinguality of my transformer model and how could I improve that // Seemingly, sBERT is way better than anything else. Tried with last layer of multilingual bert model by feedfowarding it. Results were bad.
- Check every multilingual models on SentenceTransformers --> that's the solution // that wasn't THE solution, but it's probably leading to...

[IN PROGRESS]

- Reduce the complexity: No META_ACA; only ACAClassifiers on different sbert models, keep only the best one // should be a good solution

[TODO]

- NN : check RNN
- Try to have another classifier model in parrallel which classifies directly with tokens //maybe RNN should be ok for this


[COMMENTS]

- "distiluse-base-multilingual-cased" seems to lack a bit of dimensionality: Need to do some deeper research on a at-least 768 dimensions multilingual model that is capable of transforming sentences...
- Grid Searching --> search parameters for each model, then soft voting clf // xgboost & NN grids ???
- Random Searching instead of Grid to gain time; nonetheless it seems to be way more efficient; maybe, having a strong-learners ensemble might be weaker than a waeak-learners ensemble
- The goal now is to check RNN and have a better embeddings transformer! // no better embeddings transformers than sbert
- Though, I'll check how SVC behaves with grid-searching on embeddings, might be interesting... // seems to be good in a VotingCLF, might interfer with the MLP searching
- BERT Transformer: https://metatext.io/models/bert-base-multilingual-cased // not good at all --> really need a sentence-transformer
- Try big sample sizes with hard voting classifier // also not really good
- 