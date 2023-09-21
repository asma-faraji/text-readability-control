import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import math
from readability import Readability
from scipy import stats

class ReadabilityEvaluationMetric:

    def __init__(self):
        pass

    def correlation(self, original_text_score, paraphrased_text_score):

        res = stats.spearmanr(original_text_score, paraphrased_text_score)
        corr = res.statistic

        if math.isnan(corr):
            corr = 0
        return corr

    def calculate_mse_score(self, original_text_score, paraphrased_text_score):

        MSE = np.square(np.subtract(original_text_score,paraphrased_text_score)).mean()

        return math.sqrt(MSE)

    def calculate_flesch_score(self, text):

        text = text.lower()
        r = Readability(text)

        return r.flesch().score

    def create_dataset_and_calc_scores(self, dataset):
        flesch_range = ['5', '20', '40', '55', '65', '75', '85', '95']
        final_dataset = []


        zeros = 0

        print('start calculating for 5 : ')
        final_dataset_scores = [ [self.calculate_flesch_score(data['paraphrase'])] for data in  dataset['5']]

        original_final_dataset_scores = [ [self.calculate_flesch_score(data['original'])] for data in  dataset['5']]
        
        for i in flesch_range[1:]:
            for j, data in enumerate(dataset[i]):
                print(i , j)
                # print(data['paraphrase'])
                final_dataset_scores[j].append(self.calculate_flesch_score(data['paraphrase']))
                # print(data['original'])
                original_final_dataset_scores[j].append(self.calculate_flesch_score(data['original']))
        
        with open(base_path + 'original.json', 'w') as f:
            json.dump(original_final_dataset_scores, f)

        with open(base_path + 'paraphrase.json', 'w') as f:
            json.dump(final_dataset_scores, f)

    def accuracy(self, score_list):

        is_in_range = []
        class_range = [(0, 10),(10,30), (30, 50),
                        (50, 60), (60, 70), (70,80), 
                        (80, 90), (90,100)]

        for i , _range in zip(score_list, class_range):
            is_in_range.append(int( _range[0] <= i < _range[1]))
        
        # print(is_in_range)
        return np.sum(is_in_range) / len(is_in_range)   , is_in_range  

    def get_ranking_score(self, original_final_dataset_scores, final_dataset_scores):

        flesch_range_num = [5, 20, 40, 55, 65, 75, 85, 95]
        correlation_score = []

        for score_list in final_dataset_scores:
            # print(score_list)
            corr = self.correlation(flesch_range_num, score_list)
            correlation_score.append(corr)

        print('mean corr', np.mean(correlation_score))


        # ### plot dist
        # sns.set_theme()
        # kde_plot = sns.displot(correlation_score)
        # fig = kde_plot.figure
        # fig.savefig(base_path + "corr_dist.png")

    def get_rmse_score(self, original_final_dataset_scores, final_dataset_scores):

        flesch_range_num = [5, 20, 40, 55, 65, 75, 85, 95]
        ## rmse
        par_rmse_scores = []
        for score_list in final_dataset_scores:
            rmse = self.calculate_mse_score(np.array(flesch_range_num), score_list)
            par_rmse_scores.append(rmse)

        original_rmse_scores = []
        for score_list in original_final_dataset_scores:
            rmse = self.calculate_mse_score(np.array(flesch_range_num), score_list)
            original_rmse_scores.append(rmse)


        sample_len = len(original_rmse_scores) 
        original_df = pd.DataFrame({'rmse' : original_rmse_scores, 'approach' : ['copy'] * sample_len})
        par_df = pd.DataFrame({'rmse' : par_rmse_scores, 'approach' : ['llama'] * sample_len})
        rmse_df = pd.concat([original_df, par_df], axis=0)
        print(len(rmse_df), sample_len)

        # sns.set_theme()
        # kde_plot = sns.displot(data=rmse_df, x='rmse', hue='approach',stat='probability')
        # plt.title('Distribution of rmse')
        # plt.legend()
        # plt.xlabel('rmse')
        # fig = kde_plot.figure
        # fig.savefig(base_path + "rmse_dist.png")

        print('original rmse mean : ', np.mean(original_rmse_scores))
        print('paraphrase rmse mean : ', np.mean(par_rmse_scores))

    def get_accuracy_metric(self, original_final_dataset_scores, final_dataset_scores):
        
        par_accuracy = []
        per_class_accuraccy = []
        for score_list in final_dataset_scores:
            # print(score_list)
            accuracy_score, is_in_range = self.accuracy(score_list)
            per_class_accuraccy.append(is_in_range)
            par_accuracy.append(accuracy_score)


        per_class_accuraccy = np.array(per_class_accuraccy)
        print(per_class_accuraccy.shape)
        for i in range(0, 8):
            print(np.mean(per_class_accuraccy[:, i]))

        # acc_df = pd.DataFrame({'accuracy' : par_accuracy})
        # sns.set_theme()
        # kde_plot = sns.displot(data=acc_df, x='accuracy',stat='probability')
        # plt.title('Distribution of accuracy', size=16)
        # plt.legend()
        # plt.xlabel('accuracy')
        # fig = kde_plot.figure
        # fig.savefig(base_path + "accuracy_dist.png")

        print('acc mean : ', np.mean(par_accuracy))
        print('acc std : ', np.std(par_accuracy))


    def get_regression_line(sefl, df):
        
        cols = df.columns[:-1]
        for i in cols:
            print(i)
            X = np.array(df['orig'])
            y = np.array(df[i])
            slope, intercept, r_value, p_value, std_err = stats.linregress(X,y)
            print('slope : ', slope, ' , intercept : ', intercept, 'r-squared :' , r_value**2)


    def get_df(self, original_final_dataset_scores, final_dataset_scores):
        
        flesch_range_num = [5, 20, 40, 55, 65, 75, 85, 95]

        x = []
        y = []

        red_score_dict = { 'par_' + str(i) : [] for i in flesch_range_num}
        print(red_score_dict)



        red_score_dict_keys = list(red_score_dict.keys())
        print(red_score_dict_keys)
        print(len(red_score_dict_keys))


        for i in range(0,len(flesch_range_num)):
            for a in final_dataset_scores:
    #             print(a[i], red_score_dict_keys[i])
                red_score_dict[red_score_dict_keys[i]].append(a[i])


        red_score_dict['orig'] = []
        for i in original_final_dataset_scores :
            red_score_dict['orig'].append(i[0])


        
        df = pd.DataFrame(red_score_dict)

        return df
        


def main():

    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    parser.add_argument('--data_path', type=str, help='path to the paraphrase result of llama or chatgpt')
    args = parser.parse_args()
    base_path = args.data_path

    print(base_path)
    evaluator = ReadabilityEvaluationMetric()
    origin_score = []
    paraphrase_score = []

    data_path = base_path + '{file_name}'
    flesch_range = ['5', '20', '40', '55', '65', '75', '85', '95']
    dataset = {}
    final_dataset = []

    for r in flesch_range:
        score_range = r
        file_name = 'para_' + r + '.json'
        print(data_path.format(file_name=file_name))
        dataset[r] = json.load(open(data_path.format(file_name=file_name)))



    evaluator.create_dataset_and_calc_scores(dataset)

    f  = open(base_path + 'original.json')
    original_final_dataset_scores = json.load(f)

    f  = open(base_path + 'paraphrase.json')
    final_dataset_scores = json.load(f)

    evaluator.get_ranking_score(original_final_dataset_scores, final_dataset_scores)
    evaluator.get_rmse_score(original_final_dataset_scores, final_dataset_scores)
    evaluator.get_accuracy_metric(original_final_dataset_scores, final_dataset_scores)

if __name__ == '__main__':
    main()