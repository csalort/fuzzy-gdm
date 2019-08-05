import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.stats import norm
import numpy as np
from scipy import stats


def plot_simulation_results(path, plot_p):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    d = pd.DataFrame(data={'over': data['over'], 'predict': data['predict']})
    group = 'over'
    column = 'predict'
    grouped = d.groupby(group)
    names, vals, xs = [], [], []
    for i, (name, subdf) in enumerate(grouped):
        names.append(name)
        vals.append(subdf[column].tolist())
        xs.append(np.random.normal(i + 1, 0.04, subdf.shape[0]))
    plt.figure('Simulation')
    plt.boxplot(vals, labels=names)
    plt.title('Risk simulation')
    plt.xlabel('Values over the threshold')
    plt.ylabel('Associated risk')
    plt.hlines(45, 0, 30, alpha=0.2, color='blue')
    plt.vlines(6.5, 10, 100, alpha=0.2, color='blue')
    plt.savefig(plot_p + 'simulation.jpg', format='png', figsize=(16, 16), dpi=200)


def borderline_cases(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data={'over': data['over'], 'predict': data['predict']})
    f_low_risk = ((df['over'] >= 8) & (df['predict'] <= 45))
    data_low_risk = [x for i, x in enumerate(data['data']) if f_low_risk[i]]
    f_high_risk = ((df['over'] < 8) & (df['predict'] >= 45))
    data_high_risk = [x for i, x in enumerate(data['data']) if f_high_risk[i]]
    post_low = np.concatenate([el[0:21] for el in data_low_risk])
    pre_low = np.concatenate([el[21:28] for el in data_low_risk])
    post_high = np.concatenate([el[0:21] for el in data_high_risk])
    pre_high = np.concatenate([el[21:28] for el in data_high_risk])
    post = stats.ks_2samp(post_low, post_high)
    pre = stats.ks_2samp(pre_low, pre_high)
    print('KS p-value for post: {}'.format(post[1]))
    print('KS p-value for pre: {}'.format(pre[1]))
    post_low_n = norm.fit(post_low)
    post_high_n = norm.fit(post_high)
    print('Mean for post low: {}, mean for post high: {}'.format(post_low_n[0], post_high_n[0]))
    pre_low_n = norm.fit(pre_low)
    pre_high_n = norm.fit(pre_high)
    print('Mean for pre low: {}, mean for post high: {}'.format(pre_low_n[0], pre_high_n[0]))


if __name__ == '__main__':
    plot_path = '../plots/'

    simulation_path = '../data/simdata_real.pkl'
    plot_simulation_results(simulation_path, plot_path)
    borderline_cases(simulation_path)

    original_data_path = '../data/clean_data.csv'
    # plot_original_data(original_data_path, plot_path)
