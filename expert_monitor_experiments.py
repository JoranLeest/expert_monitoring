import numpy as np
import pandas as pd
import random
import yaml
import os
import arviz as az
from multiprocessing import Process
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import ks_2samp
from scipy.special import gammaln
import matplotlib.pyplot as plt
import pymc as pm
import seaborn as sns
from matplotlib import cm
from scipy.interpolate import griddata

import warnings
warnings.filterwarnings("ignore")

os.environ["PYTHONIOENCODING"] = "'utf-8'"

def sigma_to_precision(sigma):
    return 1 / sigma**2

def precision_to_sigma(precision):
    variance = 1 / precision
    return np.sqrt(variance)
    
def expert_mean_sigma_to_alpha_beta(mean, sigma):
    alpha = mean**2 / sigma**2
    beta = mean / sigma**2
    return alpha, beta

def expert_estimate_to_alpha_beta(mean_estimate, sigma_prop_of_mean, precision=True):
    if precision:
        mean_estimate = sigma_to_precision(mean_estimate)
        
    sigma_estimate = sigma_prop_of_mean * mean_estimate
    
    alpha = mean_estimate**2 / sigma_estimate**2
    beta = mean_estimate / sigma_estimate**2
    return alpha, beta

class ExpertMonitor(object):
    def __init__(self,
            expert_model_specs,
            monitor_config = {
                'signficance_threshold': 0.05,
                'bayes_factor_threshold': 1.0,
                'window_size': 30,
                'reference_uncertainty': 0.1,
                'grace_period': 500
            }
        ):
        
        self.signficance_threshold = monitor_config['signficance_threshold']
        self.expert_model_specs = self.compile_specs(expert_model_specs)
        self.bayes_factor_threshold = monitor_config['bayes_factor_threshold']
        self.window_size = monitor_config['window_size']
        self.reference_uncertainty = monitor_config['reference_uncertainty']
        self.current_model = None
        self.predictor_class = RandomForestRegressor
        self.predictor = None
        self.grace_period = monitor_config['grace_period']
    
    def compile_specs(self, specs):
        for model_spec in specs.values():
            for feature_name, feature_params in model_spec.items():
                alpha, beta = expert_estimate_to_alpha_beta(
                    model_spec[feature_name]['std']['mu'],
                    model_spec[feature_name]['std']['sigma']
                )
                
                model_spec[feature_name]['std']['alpha'] = alpha
                model_spec[feature_name]['std']['beta'] = beta

        return specs
    
    def train_predictor(self, data):
        x = data.drop(['target', 'domain'], axis=1)
        y = data['target']
        
        self.predictor = self.predictor_class()
        self.predictor.fit(x, y)
    
    def monitor_offline(self, data):
        model_inputs = data.drop(['target', 'domain'], axis=1)
        in_grace = True
        in_grace_instances = 0
        training_data_indices = [None, None]
        current_domain = 'base'
        detected_domains = []
        true_domains = []
        predictions = []
        for i, observation in model_inputs.iterrows():
            
            if in_grace_instances == 0 and in_grace:
                training_data_indices[0] = i
            elif in_grace_instances == (self.grace_period - 1) and in_grace:
                training_data_indices[1] = i
                in_grace = False
                in_grace_instances = 0
                
                self.train_predictor(data.iloc[training_data_indices[0]:training_data_indices[1], :])
                predictions = []
            elif in_grace == False:
                current_domain = data.loc[i, 'domain']
                
                predictions.append(self.predictor.predict([model_inputs.loc[i,:]])[0])
                
                if len(predictions) >= self.window_size * 2:
                    result = self.test(predictions[:i], data.iloc[:i,:])
                    
                    if result['drift_detected']:
                        # print(f'CURRENT DOMAIN: {current_domain}')
                        detected_domains.append(result['drift_detected'])
                        true_domains.append(current_domain)
                        
                        in_grace = True
                    
            in_grace_instances += 1

        return accuracy_score(detected_domains, true_domains)
    
    def test(self, predictions, data):        
        test_reference = predictions[-self.window_size:]
        test_observed = predictions[2*-self.window_size:-self.window_size]   

        st, p_value = ks_2samp(test_reference, test_observed, method='auto')
        
        drift_detected = p_value < self.signficance_threshold # and st > 0.1
        
        detected_scenario = None
        if drift_detected:
            data.pop('target')
            data.pop('domain')
            
            reference_data, observed_data = [{} for i in range(2)]
            for feature_name, feature_values in data.items():
                reference_data[feature_name] = feature_values[-self.window_size:]
                observed_data[feature_name] = feature_values[2*-self.window_size:-self.window_size]      
        
            detected_scenario = self.compare(reference_data, observed_data)
            
        return {'drift_detected': drift_detected, 'detected_scenario': detected_scenario}
 
    def compare(self,
            reference_data,
            observed_data,
            rng=42,
            num_samples=10,
            approx_method='conjugate'
        ):
        
        alpha_betas = {feature_name: {} for feature_name in reference_data.keys()}
        for feature_name in reference_data.keys():
            alpha_ref, beta_ref = expert_estimate_to_alpha_beta(
                np.std(reference_data[feature_name]),
                self.reference_uncertainty
                # max(0.1, 0.5 - (0.005 * len(reference_data[feature_name])))
            )
            
            alpha_betas[feature_name]['alpha'] = alpha_ref
            alpha_betas[feature_name]['beta'] = beta_ref
            
        reference_model = {
            feature_name: {
                "mean": {
                    "mu": np.mean(reference_data[feature_name]),
                    "sigma": max(0.01, 0.5 - (0.005 * len(reference_data[feature_name])))
                },
                "std": {
                    "alpha": alpha_betas[feature_name]['alpha'],
                    "beta": alpha_betas[feature_name]['beta']
                }
            } for feature_name in reference_data.keys()
        }
        
        if approx_method == 'MCMC':        
            models, traces = [{} for i in range(2)]
            for model_name, model_spec in {**self.expert_model_specs, **{"reference": reference_model}}.items():
                with pm.Model() as combined_model:
                    for feature_name, feature_observed in observed_data.items():
                        mean = pm.Normal(
                            f'mean_{feature_name}',
                            mu=model_spec[feature_name]['mean']['mu'],
                            sigma=model_spec[feature_name]['mean']['sigma']
                        )
                        
                        sd = pm.InverseGamma(
                            f'sd_{feature_name}',
                            alpha=model_spec[feature_name]['std']['alpha'],
                            beta=model_spec[feature_name]['std']['beta']
                        )
                
                        likelihood = pm.Normal(
                            feature_name,
                            mu=mean,
                            sigma=sd,
                            observed=feature_observed
                        )
                    
                    models[model_name] = combined_model     
                    
                    traces[model_name] = pm.sample(
                        100,
                        tune=100,
                        chains=2, 
                        cores=1,
                        progressbar=False
                    )
                    
                    pm.compute_log_likelihood(traces[model_name])

            model_comparison = az.compare(traces)
            best_model = model_comparison['elpd_diff'].idxmax()
            top_diff = model_comparison.loc['reference', 'elpd_diff']
    
            if top_diff > self.bayes_factor_threshold:
                self.current_model = best_model
            
                return best_model
            else:
                return None
            
        elif approx_method == 'conjugate':
            model_marginal_likelihoods = {}
            for model_name, model in {**self.expert_model_specs, **{"reference": reference_model}}.items():
                feature_marginal_likelihoods = []
                for feature_name, feature_observed in observed_data.items():
                    mu0 = model[feature_name]['mean']['mu']  # prior mean
                    kappa0 = sigma_to_precision(model[feature_name]['mean']['sigma']) # prior precision
                    alpha0 = model[feature_name]['std']['alpha'] 
                    beta0 = model[feature_name]['std']['beta'] 
                    
                    # Compute the posterior hyperparameters
                    n = len(feature_observed)
                    mu_n = (kappa0 * mu0 + n * np.mean(feature_observed)) / (kappa0 + n)
                    kappa_n = kappa0 + n
                    alpha_n = alpha0 + n / 2
                    beta_n = beta0 + 0.5 * np.sum((feature_observed - np.mean(feature_observed)) ** 2) + (
                            kappa0 * n * (np.mean(feature_observed) - mu0) ** 2) / (2 * (kappa0 + n))
                   
                    # Compute the logarithm of the gamma function
                    log_gamma_alpha_n = gammaln(alpha_n)
                    log_gamma_alpha0 = gammaln(alpha0)
                   
                    # Compute the logarithm of the marginal likelihood
                    log_marginal_likelihood = (log_gamma_alpha_n - log_gamma_alpha0) + \
                                              (alpha0 * np.log(beta0) - alpha_n * np.log(beta_n)) + \
                                              0.5 * (np.log(kappa0) - np.log(kappa_n)) - \
                                              (n / 2) * np.log(2 * np.pi)
                   
                    # marginal_likelihood = np.exp(log_marginal_likelihood)
           
                    feature_marginal_likelihoods.append(log_marginal_likelihood)
    
                model_marginal_likelihood = np.sum(feature_marginal_likelihoods)
                model_marginal_likelihoods[model_name] = model_marginal_likelihood
            
            bayes_factors = {}
            for model_name, evidence in model_marginal_likelihoods.items():
                if model_name != 'reference':
                    bayes_factors[model_name] = evidence - model_marginal_likelihoods['reference']
            
            # print(bayes_factors)

            if bayes_factors:
                best_model = max(bayes_factors, key=bayes_factors.get)
                if bayes_factors[best_model] > self.bayes_factor_threshold:                    
                    return best_model
                else:
                    return None
            else:
                return None
            
class ResourceGenerator(object):
    def __init__(self,
        specification = {
            'num_samples': 1000,
            'num_features': 10,
            'num_domains': 40,
            'num_models': 10,
            'max_mean': 100,
            'drift_step': 0.2,
            'max_std': 20,
            'estimate_uncertainty': 0.1,
            'estimate_error': 0.01
        },
        rng = 42
        ):
        self.specification = specification
        self.rng = rng

    def gen_domain_models(self, domains):
        np.random.seed(self.rng)
        random.seed(self.rng)
        
        selected_domains = dict(random.sample(list(domains.items()), self.specification['num_models']))
        
        domain_models = {}
        for i, (domain_name, domain_params) in enumerate(selected_domains.items(), start=1):
            domain_models[domain_name] = {
                feature_name: {
                    'mean': {
                        'mu': feature_params[0] * (1 + (np.random.choice([-1, 1]) * self.specification['estimate_error'])),
                        'sigma': self.specification['estimate_uncertainty'] # feature_params[0] * self.specification['estimate_uncertainty']
                    },
                    'std': {
                        'mu': feature_params[1] * (1 + (np.random.choice([-1, 1]) * self.specification['estimate_error'])),
                        'sigma': self.specification['estimate_uncertainty'] # feature_params[1] * self.specification['estimate_uncertainty']
                    }
                } for feature_name, feature_params in domain_params.items()
            }
            
        return domain_models

    def gen_drifted_stream(self):
        np.random.seed(self.rng)
        random.seed(self.rng)
        
        base_stream, base_params = self.gen_base_features()
        base_stream['target'] = self.gen_target(base_stream)
        
        data_stream = pd.DataFrame(base_stream)
        data_stream['domain'] = 0
        
        domains = {}
        for i in range(1, self.specification['num_domains']):
            drift_stream, base_params = self.gen_drift(base_params)
            drift_stream['target'] = self.gen_target(drift_stream)
            
            drift_stream = pd.DataFrame(drift_stream)
            drift_stream['domain'] = i
            
            data_stream = pd.concat([data_stream, drift_stream]).reset_index(drop=True)
            
            domains[i] = base_params
   
        return (data_stream, domains)

    def gen_target(self, synth_features):
        np.random.seed(self.rng)
        random.seed(self.rng)
        
        target = 0
        for synth_feature in synth_features.values():
            target += np.random.choice(np.arange(0.1, 6, 10))*synth_feature
                                       
        target += np.random.normal(5, 1, self.specification['num_samples'])
        
        return target

    def gen_base_features(self):
        np.random.seed(self.rng)
        random.seed(self.rng)
        
        feature_params, synth_features = [{} for i in range(2)]
        for i in range(self.specification['num_features']):
            random_mean = np.random.choice(np.arange(20, self.specification['max_mean'], 10))
            random_std = np.random.choice(np.arange(5, self.specification['max_std'], 10))
            generated_stream = np.random.normal(
                random_mean,
                random_std,
                self.specification['num_samples']
            )
            
            feature_params[f'feature_{i}'] = (random_mean, random_std)
            synth_features[f'feature_{i}'] = generated_stream
        
        return (synth_features, feature_params)
        
    def gen_drift(self, feature_params):
        np.random.seed(self.rng)
        random.seed(self.rng)
        
        step = self.specification['drift_step']
        
        new_feature_params, new_synt_features = [{} for i in range(2)]
        for i, (feature, params) in enumerate(feature_params.items()):
            change = np.random.choice([1, 1 - step, 1 + step])

            new_mean = params[0] * change
            new_std = params[1] * change
            
            generated_stream = np.random.normal(
                new_mean,
                new_std,
                self.specification['num_samples']
            )
            
            new_feature_params[f'feature_{i}'] = (new_mean, new_std)
            new_synt_features[f'feature_{i}'] = generated_stream
        
        return new_synt_features, new_feature_params

def oracle_eval(monitor, data, num_drifts, window_size):
    slices = []
    for i in range(num_drifts):
        start_row = i * window_size
        end_row = start_row + window_size
        df_slice = data.iloc[start_row:end_row].reset_index(drop=True)
        slices.append(df_slice)

    last_slice = pd.DataFrame()
    detected_domain, current_domain = [0 for _ in range(2)]
    detected_domains, true_domains = [[] for _ in range(2)]
    for new_slice in slices:

        current_domain = new_slice.loc[0, 'domain']
        
        new_slice.pop('target')
        new_slice.pop('domain')
        
        if len(last_slice) > 0:
            detected_domain = monitor.compare(last_slice, new_slice)
            
            detected_domains.append(detected_domain)
            true_domains.append(current_domain) 
            
        last_slice = new_slice
        
    filtered_detected_domains = [d for d in detected_domains if d is not None]
    filtered_true_domains = [t for d, t in zip(detected_domains, true_domains) if d is not None]
    
    return accuracy_score(filtered_detected_domains, filtered_true_domains)

def experiment(
        config,
        iterations=None,
        rng=None,
        filename=None,
        threads=None
    ):

    config, iterations, rng, filename, threads = [
        arg_value or config[arg_name] for arg_name, arg_value in locals().items()
    ]

    if threads > 1:
        parallel_experiments(
            *(var for var in locals().values())
        )
    else:
        estimate_uncertainties = config['estimate_uncertainties']
        estimate_errors = config['estimate_errors']
        
        results = {column: [] for column in ['errors', 'uncertainties', 'accuracy', 'iteration']}
        for uncertainty in estimate_uncertainties:
            for error in estimate_errors:

                config['generator_specification']['estimate_uncertainty'] = uncertainty
                config['generator_specification']['estimate_error'] = error
                
                for i in range(iterations):
                    generator = ResourceGenerator(config['generator_specification'], rng=rng + 1)
                    
                    data, domains = generator.gen_drifted_stream()
                    models = generator.gen_domain_models(domains)
                    
                    xMonitor = ExpertMonitor(
                        expert_model_specs = models,
                        monitor_config = config['monitor_config']
                    )
                    
                    accuracy = oracle_eval(
                        xMonitor,
                        data, 
                        config['generator_specification']['num_domains'],
                        config['monitor_config']['window_size']
                    )
                    
                    results['errors'].append(error)
                    results['uncertainties'].append(uncertainty)
                    results['accuracy'].append(accuracy)
                    results['iteration'].append(i)
                    
        pd.DataFrame(results).to_csv(f'results/{filename}.csv', index=False)
            
        return results

def parallel_experiments(
        config,
        iterations=None,
        rng=None,
        filename=None,
        threads=None
    ):

    config, iterations, rng, filename, threads = [
        arg_value or config[arg_name] for arg_name, arg_value in locals().items()
    ]
    
    iters_per_process = int(iterations / threads)
    
    global experiment
    processes = [
        Process(
            target=experiment,
            args=(
                config,
                iters_per_process,
                rng + (i * iters_per_process),
                f'{filename}-{i + 1}',
                1
            )
        ) for i in range(threads)
    ]
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()

    x_results = pd.DataFrame()
    for i in range(1, threads + 1):
        sub_results = pd.read_csv(f'results/{filename}-{i}.csv')
        os.remove(f'results/{filename}-{i}.csv')
        
        sub_results['iteration'] = i
        
        x_results = pd.concat([x_results, sub_results]).reset_index(drop=True)
        
    x_results.to_csv(f'results/{filename}.csv', index=False)
    
    return x_results
    
def surface_plot(filename):
    data = pd.read_csv(f'{filename}.csv')
    
    data = data.fillna(0)
    
    data = data.groupby(['errors', 'uncertainties'])['accuracy'].mean().reset_index()

    data = data[(data.uncertainties <= 0.25) & (data.errors <= 0.25)]

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    
    x = np.array(data['errors'].unique())
    z = np.array(data['uncertainties'].unique())
    
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Create meshgrid
    X, Z = np.meshgrid(x, z)
    
    # Interpolate the data to create a smooth surface
    points = data[['errors', 'uncertainties']].values
    values = data['accuracy'].values
    interp_x = np.linspace(x.min(), x.max(), 100)
    interp_z = np.linspace(z.min(), z.max(), 100)
    interp_X, interp_Z = np.meshgrid(interp_x, interp_z)   
    interp_Y = griddata(points, values, (interp_X, interp_Z), method='cubic')

    plane_data = ax.plot_surface(
        interp_X,
        interp_Z,
        interp_Y,
        cmap=cmap,
        alpha=1,
        shade=True,
        linewidth=5,
        antialiased=False
    )
    
    plane_data = ax.plot_wireframe(interp_X, interp_Z, interp_Y, rstride=3, cstride=3, color='black', linewidth=0.4)
    
    
    scale = 0.65  # Adjust this value as needed
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, scale, 1]))
    
    z = points[:, 1]
    X, Z = np.meshgrid(np.unique(x), np.unique(z))
    
    # Find the corresponding values for each point in the meshgrid
    Y = np.zeros_like(X)
    for i in range(len(x)):
        xi = np.where(np.unique(x) == x[i])[0][0]
        zi = np.where(np.unique(z) == z[i])[0][0]
        Y[zi, xi] = values[i]
    
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 4
    
    X, Z = data['errors'].values, data['uncertainties'].values
    values = data['accuracy'].values
    bottom = np.array([min(values) * 1.0001] * len(X))
    width = depth = 0.025
    
    colors = cm.plasma((values - bottom) / (np.max(values) - bottom))
    
    # Add colorbar
    cbar = fig.colorbar(plane_data, shrink=0.37, aspect=10, cmap=cmap)
    cbar.ax.tick_params(labelsize=14)
    
    # Set labels for axes
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Estimate error', fontsize=16)
    ax.set_ylabel('Estimate uncertainty', fontsize=16)
    ax.set_zlabel('accuracy', fontsize=16)
    # ax.set_xlim([z.max(), z.min()])
    # ax.set_ylim([z.max(), z.min()])
    
    # Set title for plot
    ax.set_title(f'Expert Monitor - accuracy vs. estimate error and uncertainty', fontsize=16)
    
    # Set grid lines
    ax.xaxis._axinfo['grid'].update({'linewidth':0.5, 'color':'gray'})
    ax.yaxis._axinfo['grid'].update({'linewidth':0.5, 'color':'gray'})
    ax.zaxis._axinfo['grid'].update({'linewidth':0.5, 'color':'gray'})
    
    # Adjust lighting and shading
    ax.view_init(30, 220)
    ax.dist = 12
    ax.view_init(elev=20, azim=40)

if __name__ == '__main__':
    with open('configs/experiment-A04.yaml') as f:
        config = yaml.safe_load(f)

    results = experiment(config)
    
    data = surface_plot("results/experiment_A04_results")


