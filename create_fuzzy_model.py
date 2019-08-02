from skfuzzy import control as ctrl
import numpy as np
import skfuzzy as fuzz
import pickle
import matplotlib.pyplot as plt


def plot_membership(rule, path, title, name, xlabel, ylabel):
    rule.view()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if name == 'membership_risk':
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(path + name + '.jpg', format='jpg', figsize=(16, 16), dpi=200, bbox_inches='tight')


def create_variables(n_pre, n_post, limit_post, limit_pre,
                     min_universe=0, max_universe=15, precision_universe=.1, fuzzy_range=.3,
                     plt_membership=None):
    print('Creating the variables for the fuzzy system')
    r = np.arange(min_universe, max_universe, precision_universe)
    inputs = {}
    for i in range(n_post):
        v_name = 'gl_post_' + str(i)
        inputs[v_name] = ctrl.Antecedent(r, v_name)
        inputs[v_name]['normal'] = fuzz.trapmf(r, [0, 0, limit_post - fuzzy_range, limit_post + fuzzy_range])
        inputs[v_name]['high'] = fuzz.trapmf(r, [limit_post - fuzzy_range, limit_post + fuzzy_range, 15, 15])
    for i in range(n_pre):
        v_name = 'gl_pre_' + str(i)
        inputs[v_name] = ctrl.Antecedent(r, v_name)
        inputs[v_name]['normal'] = fuzz.trapmf(r, [0, 0, limit_pre - fuzzy_range, limit_pre + fuzzy_range])
        inputs[v_name]['high'] = fuzz.trapmf(r, [limit_pre - fuzzy_range, limit_pre + fuzzy_range, 15, 15])
    risk_universe = np.arange(1, 101, 1)
    outputs = ctrl.Consequent(risk_universe, 'risk')
    outputs['r0'] = fuzz.trimf(risk_universe, [0, 10, 20])
    outputs['r1'] = fuzz.trimf(risk_universe, [10, 20, 30])
    outputs['r2'] = fuzz.trimf(risk_universe, [20, 30, 40])
    outputs['r3'] = fuzz.trimf(risk_universe, [40, 50, 60])
    outputs['r4'] = fuzz.trimf(risk_universe, [50, 60, 70])
    outputs['r5'] = fuzz.trimf(risk_universe, [60, 70, 80])
    outputs['r6'] = fuzz.trimf(risk_universe, [70, 80, 90])
    outputs['r7'] = fuzz.trimf(risk_universe, [80, 90, 100])
    outputs['r8'] = fuzz.trimf(risk_universe, [90, 100, 100])
    if plot_membership is not None:
        plot_membership(inputs['gl_post_0'], plt_membership, 'Membership level of postprandial ', 'membership_post',
                        'Glucose level', 'Membership level')
        plot_membership(inputs['gl_pre_0'], plt_membership, 'Membership level of fasting ', 'membership_fast',
                        'Glucose level', 'Membership level')
        plot_membership(outputs, plt_membership, 'Membership of risk level', 'membership_risk',
                        'Risk', 'Membership level')
    return inputs, outputs


def create_rules(inputs, outputs, n_pre, n_post):
    print('Creating the rules for the fuzzy system')
    level_map = {'0': 'normal', '1': 'high'}
    post_rules = []
    thresholds = [.08, .16, .24, .28, .32, .4, .48, .54]
    for i in range(n_post + 1):
        post_rules.append((list(np.ones(i, dtype=int)) + list(np.zeros(n_post - i, dtype=int))))
    pre_rules = []
    for i in range(n_pre + 1):
        pre_rules.append((list(np.ones(i, dtype=int)) + list(np.zeros(n_pre - i, dtype=int))))
    rules_numeric = []
    for i in range(len(post_rules)):
        for j in range(len(pre_rules)):
            rules_numeric.append(post_rules[i] + pre_rules[j])
    all_rules = []
    rule_counter = 0
    for rule in rules_numeric:
        temp_sum = sum(rule)
        rule_res = 'r8'
        for el in range(len(thresholds) - 1, -1, -1):
            if temp_sum <= int(thresholds[el] * (n_pre + n_post) + 1):
                rule_res = 'r' + str(el)
        temp_rule = '('
        for index, key in enumerate(inputs):
            temp_rule += "inputs['" + key + "']['" + level_map[str(rule[index])] + "'] & "
        temp_rule = temp_rule[:-3] + ')'
        all_rules.append(ctrl.Rule(antecedent=eval(temp_rule),
                                   consequent=outputs[rule_res] % (1 / (int(rule_res[-1:]) + 1)),
                                   label='rule'+str(rule_counter)))
        print((1 / (int(rule_res[-1:]) + 1)))
        rule_counter += 1
    return all_rules


if __name__ == '__main__':
    post_meal_limit = 7
    fasting_limit = 5
    n_days_history = 7
    daily_post_measurements = 3
    model_path = '../model/gdm_ctrl.pkl'

    total_post_measurements = daily_post_measurements * n_days_history
    total_pre_measurements = n_days_history
    n_inputs = total_pre_measurements + total_post_measurements

    inputs, output = create_variables(total_pre_measurements,
                                      total_post_measurements,
                                      post_meal_limit,
                                      fasting_limit,
                                      plt_membership='../plots/')
    rules = create_rules(inputs, output,
                         total_pre_measurements, total_post_measurements)
    print('Preparing control system')
    gdm_ctrl = ctrl.ControlSystem(rules=rules)
    print('Writing to disk')
    with open(model_path, 'wb') as o:
        pickle.dump(gdm_ctrl, o, pickle.HIGHEST_PROTOCOL)
    print('Model ready, and saved into {}'.format(model_path))
