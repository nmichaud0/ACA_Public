import itertools
import collections
import pandas as pd
import numpy as np
# from rich import print
import os
import plotly.express as px
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings
import time
warnings.simplefilter('error')
startTime = time.time()

# Goal here is to find an ensemble of models that have the best performances
#
# The strategy is to brute-force the data: by defining combinations (without replacement) of models iteratively and
# test them with soft and hard voting on the first half of the predictions
#
# When we have an ensemble that have a better balanced_accuracy than its components we save it
# Then we re-test the savec ensembles on the other half of the test set and we'll take a portion of the best to
# further evaluate depending on their first performances
#
# Will need to randomize the two halves of testing ! // like a bootstrap \\ OK
# TODO: Evaluate accuracy with only strong probabilities // cross evaluation of this method
# TODO: Consider using a ML algorithm to find the bests models combinations instead of brute-forcing

true_values_path = '/Users/nizarmichaud/PycharmProjects/ACA_Public/' \
                   'B_ARCH_output/true_values_for_crossmodeleval_transposed.xlsx'
models_acc_path = '/Users/nizarmichaud/PycharmProjects/ACA_Public/B_ARCH_output/cross-models_evaluation.xlsx'
models_prob_path = '/Users/nizarmichaud/PycharmProjects/ACA_Public/B_ARCH_output/prob_for_crossmodeleval_halved.xlsx'

df_true_val = pd.read_excel(true_values_path)
df_models_acc = pd.read_excel(models_acc_path)
df_models_prob = pd.read_excel(models_prob_path)

sbert_models = ['distiluse-base-multilingual-cased-v1',
                'distiluse-base-multilingual-cased-v2',
                'paraphrase-multilingual-MiniLM-L12-v2',
                'paraphrase-multilingual-mpnet-base-v2',
                'all-mpnet-base-v2',
                'multi-qa-mpnet-base-dot-v1',
                "gtr-t5-large",
                "multi-qa-mpnet-base-cos-v1"]

models = ['xgboost', 'rf', 'mlp', 'mlps', 'svc', 'svcs', 'gnb', 'lr', 'kn']

every_models = {f'{model_label}-{sb_label}': (sb_label, model_label)
                for model_label in models
                for sb_label in sbert_models}

models_list = list(every_models.keys())

assert models_list == df_models_acc['model'].tolist()


def comb(models_local: list, n: int):

    """
    Returns all the model combinations list for all the models we consider: models_local and for sample of size n
    :param models_local:
    :param n:
    :return: combination list
    """

    return list(itertools.combinations(models_local, n))


max_comb, max_models = len(models_list), len(models_list)

# Making dic for models proba as : {model_label0: [p0, p1, ..., pn], ... m_ln: [p0, ..., pn]}

models_prob = {}
data = df_models_prob.to_dict('split')
for i in range(df_models_prob.shape[0]):
    models_prob[data['data'][i][0]] = np.asarray(data['data'][i][1:])


true_val = np.asarray(df_true_val.to_dict('split')['data'][0][1:])


def shuffle_np_arrays(t_val: np.array, prob_matrix: dict):

    """
    Shuffles the test set evenly so we have less chances of "overfitting" our test data
    Data that should be used: true_val and the models probabilities
    :param t_val:
    :param prob_matrix:
    :return: tval and probmatrix evenly shuffled
    """

    assert len(t_val) == len(list(prob_matrix.values())[0])

    perm = np.random.permutation(len(t_val))

    t_val_perm = t_val[perm]

    prob_matrix_perm = {key: val[perm] for key, val in prob_matrix.items()}

    return {'tval': t_val_perm, 'prob_matrix': prob_matrix_perm, 'perm': perm}


def soft_voting(prob_matrix: np.array):

    """
    This is a self clone of the function of soft voting of sklearn
    It returns the probability of defining class 1 or 0 depending on the mean proba of all the models
    round() is used because it understands better a 0.4999... -> 0 and 0.50...01 -> 1 than int()
    :param prob_matrix:
    :return: prob array
    """

    probs = []
    for col in range(prob_matrix.shape[1]):
        probs.append(round(np.mean(prob_matrix[:, col])))

    return np.asarray(probs)


def hard_voting(prob_matrix: np.array):

    """
    This returns the hard voting technique on a prob_matrix: computes all the classifications on each models (round(x))
    Then selects the most common value with argmax and bincount; if equal amounts, return the first one (only in even
    and not odd cases)
    :param prob_matrix:
    :return:
    """

    probs = []
    for col in range(prob_matrix.shape[1]):

        classification_local = []
        for value in prob_matrix[:, col]:

            classification_local.append(round(value))

        result = np.argmax(np.bincount(classification_local))
        probs.append(result)

    return np.asarray(probs)


def evaluate_combination(models_combination: list, shuffled_test_set: np.array, shuffled_probas: dict,
                         perm: np.array, saved_combinations: pd.DataFrame, voting: str = 'soft', iter_n: int = 0):

    models_bacc = [float(df_models_acc.loc[df_models_acc['model'] == model, 'balanced_accuracy']) for model in models_combination]

    probas_matrix = np.asarray([shuffled_probas[model] for model in models_combination])

    # splitting test set and shuffled probas to cross-evaluate

    split = int(len(shuffled_test_set)/2)

    vote_test_set = shuffled_test_set[:split]
    evaluate_test_set = shuffled_test_set[split:]

    vote_probas_matrix = probas_matrix[:, :split]
    evaluate_probas_matrix = probas_matrix[:, split:]

    if voting == 'soft':
        vote = soft_voting(vote_probas_matrix)
        eval_vote = soft_voting(evaluate_probas_matrix)
        all_vote = soft_voting(probas_matrix)
    else:  # Hard voting
        vote = hard_voting(vote_probas_matrix)
        eval_vote = hard_voting(evaluate_probas_matrix)
        all_vote = hard_voting(probas_matrix)

    vote_bacc_combination = balanced_accuracy_score(vote_test_set, vote)
    eval_bacc_combination = balanced_accuracy_score(evaluate_test_set, eval_vote)

    all_bacc = balanced_accuracy_score(shuffled_test_set, all_vote)
    all_acc = accuracy_score(shuffled_test_set, all_vote)

    # Saves the combination if the score is superior or equal to the max model
    # if vote_bacc_combination >= max(models_bacc):
    if all_bacc >= 0.8:

        # New row
        df_combination = pd.DataFrame({'iter_n': iter_n,
                                       'combination': [models_combination],
                                       'combination_bacc': [models_bacc],
                                       'vote_bacc': vote_bacc_combination,
                                       'eval_bacc': eval_bacc_combination,
                                       'all_bacc': all_bacc,
                                       'all_acc': all_acc,
                                       'voting': voting,
                                       'perm': [perm]})

        return pd.concat([saved_combinations, df_combination])

    else:
        return saved_combinations


df_combinations = pd.DataFrame({'iter_n': [float('Nan')],
                                'combination': [float('Nan')],
                                'combination_bacc': [float('Nan')],
                                'vote_bacc': [float('Nan')],
                                'eval_bacc': [float('Nan')],
                                'all_bacc': [float('Nan')],
                                'all_acc': [float('Nan')],
                                'voting': [float('Nan')],
                                'perm': [float('Nan')]})


combinations = comb(models_list, 4)

i = 0
for voting_method in ['soft', 'hard']:
    for c in combinations:
        c = list(c)
        shuffled = shuffle_np_arrays(true_val, models_prob)
        true_values_shuffled = shuffled['tval']
        models_prob_shuffled = shuffled['prob_matrix']
        perm_key = shuffled['perm']

        df_combinations = evaluate_combination(models_combination=c,
                                               shuffled_test_set=true_values_shuffled,
                                               shuffled_probas=models_prob_shuffled,
                                               perm=perm_key,
                                               saved_combinations=df_combinations,
                                               voting=voting_method,
                                               iter_n=i)
        i += 1
        """
        if i > 100:
            break
    break
        """
df_combinations.to_excel('/Users/nizarmichaud/PycharmProjects/ACA_Public/B_ARCH_output/combination_eval.xlsx')

print(f'Process took: {time.time()-startTime}')
