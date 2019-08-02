import numpy as np
import pickle
from skfuzzy import control as ctrl


def run_simulations(model, n_post, n_pre, out, data=None, n_simulations=5000, post_limit=7, fast_limit=5, seed=1993,
                    fast_mean=4., fast_u=1., post_mean=6., post_u=1., n_bigger_mean=.8):
    np.random.seed(seed)
    if data is not None:
        n_simulations = data['data'].shape[0]
    else:
        data = {'data': [], 'over': []}
        for i in range(int(n_simulations * n_bigger_mean)):
            a = np.around(np.random.normal(post_mean, post_u, n_post), decimals=1)
            b = np.around(np.random.normal(fast_mean, fast_u, n_pre), decimals=1)
            data['data'].append(sorted(list(a), reverse=True) + sorted(list(b), reverse=True))
            data['over'].append(sum(a > post_limit) + sum(b > fast_limit))
        for i in range(int(n_simulations * n_bigger_mean), n_simulations):
            a = np.around(np.random.normal(post_mean + 1, post_u, n_post), decimals=1)
            b = np.around(np.random.normal(fast_mean + 1, fast_u, n_pre), decimals=1)
            data['data'].append(sorted(list(a), reverse=True) + sorted(list(b), reverse=True))
            data['over'].append(sum(a > post_limit) + sum(b > fast_limit))

    print('Running {} simulations with the model located in {}'.format(n_simulations, model))
    print('Loading model')
    with open(model, 'rb') as o:
        gdm_ctrl = pickle.load(o)
    print('Create simulation module')
    gdm = ctrl.ControlSystemSimulation(gdm_ctrl)
    input_names = ['gl_post_' + str(i) for i in range(n_post)] + ['gl_pre_' + str(i) for i in range(n_pre)]
    data['predict'] = []
    print('Start the simulations')
    for sim in range(n_simulations):
        if (sim + 1) % 100 == 0:
            print('Predicting for step {}'.format(sim))
        gdm.inputs({input_names[var]: data['data'][sim][var] for var in range(len(input_names))})
        gdm.compute()
        data['predict'].append(gdm.output['risk'])
    with open(out, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    post_meal_limit = 7
    fasting_limit = 5
    n_days_history = 7
    daily_post_measurements = 3
    model_path = '../model/gdm_ctrl.pkl'
    output_path = '../data/'

    total_post_measurements = daily_post_measurements * n_days_history
    total_pre_measurements = n_days_history
    n_inputs = total_pre_measurements + total_post_measurements

    run_simulations(model_path, total_post_measurements, total_pre_measurements, output_path + 'simdata_real.pkl',
                    fast_mean=4.86, fast_u=0.64, post_mean=6.43, post_u=1.23)

