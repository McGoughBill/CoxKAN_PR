"""
Main module for CoxKAN class.
"""

import torch
from kan import KAN
from kan.LBFGS import LBFGS
from lifelines.utils import concordance_index
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy
from kan.utils import fit_params
import os
from tqdm import tqdm
from pathlib import Path
import uuid
from torch import Tensor
from sklearn.model_selection import train_test_split
import json


TEMP_CKPT_DIR = Path(__file__).parent / '_ckpt'
os.makedirs(TEMP_CKPT_DIR, exist_ok=True)

ordered_functions = [
    "0",        # The identically zero model: no variability.
    "1",        # A constant model: the minimal assumption if nothing varies.
    "x",        # Linear: basic direct proportionality; parameters have clear interpretation.
    "exp",      # Exponential: models constant percentage growth or decay.
    "log",  # Logarithm: compresses scale, ubiquitous in economics and biology.
    "sigmoid",  # Sigmoid (logistic): a classic S–curve used in classification.
    "gaussian",  # Gaussian: bell–shaped, symmetric; central in statistics, probability theory, and kernel methods.
    "x^2",  # Quadratic: one bend, symmetric curvature.
    "sqrt",  # Square root: a sublinear transformation; often used for variance stabilization.
    "sin",  # Sine: basic periodic oscillation.
    "1/x"  # Introduces a pole at zero
    "tan",  # Tangent: periodic with discontinuous asymptotes (implying stronger assumptions).
    ]

from .utils import FastCoxLoss, categorical_fun, Logger, SYMBOLIC_LIB


class CoxKAN(KAN):
    """
    CoxKAN class

    Attributes:
    ------------
        act_fun: a list of KANLayer
            KANLayers
        depth: int
            depth of KAN
        width: list
            number of neurons in each layer. e.g., [2,5,5,3] means 2D inputs, 3D outputs, with 2 layers of 5 hidden neurons.
        grid: int
            the number of grid intervals
        k: int
            the order of piecewise polynomial
        base_fun: fun
            residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        symbolic_fun: a list of Symbolic_KANLayer
            Symbolic_KANLayers

    Methods:
    --------
        __init__():
            initalize a CoxKAN model
        process_data():
            preprocess dataset and register metadata
        train():
            train the model
        cindex():
            compute concordance index
        predict():
            predict the log-partial hazard
        predict_partial_hazard():
            predict the partial hazard (exp of log-partial hazard)
        prune_edges():
            prune edges (activation functions) of the model
        prune_nodes():
            prune nodes (neurons) of the model
        fix_symbolic():
            set (l,i,j) activation to be symbolic (specified by fun_name)
        plot():
            plot the model
        plot_act():
            plot a specific activation function
        suggest_symbolic():
            find the best symbolic function for a specific activation (highest r2)
        auto_symbolic():
            automatic symbolic fitting
        symbolic_formula():
            obtain the symbolic formula of the full model
        symbolic_rank_terms():
            calculate standard devation of each term in symbolic formula
    """

    def __init__(self, **kwargs):
        '''
        Initalize a CoxKAN model
        
        Keyword Args:
        -----
            width : list of int
                :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            noise_scale : float
                initial injected noise to spline. Default: 0.1.
            base_fun : fun
                the residual function b(x). Default: torch.nn.SiLU().
            symbolic_enabled : bool
                compute or skip symbolic computations (for efficiency). By default: True. 
            bias_trainable : bool
                bias parameters are updated or not. By default: True
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            
        Returns:
        --------
            self
        '''
        if kwargs.get('base_fun')=='silu':
            kwargs['base_fun'] = torch.nn.SiLU()
        elif kwargs.get('base_fun')=='linear':
            kwargs['base_fun'] = torch.nn.Identity()
        self.config = dict(kwargs)
        self.pruned = False
        self.symbolic = False
        super(CoxKAN, self).__init__(**kwargs)

    def process_data(self, df_train, df_test, duration_col, event_col, covariates=None, categorical_covariates=True, normalization='minmax'):
        """
        Preprocess dataset and register metadata via the following steps:
            - Encode categorical covariates via label-encoding (if categorical_covariates is not None)
            - Normalize covariates
            - Register metadata: duration_col, event_col, covariates, normalizer, categorical_covariates and category_maps (maps from the encoded values of each category to the original names)
        
        Args:
        -----
            df_train : pd.DataFrame
                training dataset
            df_test : pd.DataFrame
                testing dataset
            duration_col : str
                column name for duration
            event_col : str
                column name for event
            covariates : list
                list of covariates. If None, all columns except duration_col and event_col are used.
            categorical_covariates : bool or list
                If True, categorical covariates are automatically detected and label encoded. 
                If a list is provided, only the covariates in the list are label encoded.
            normalization : str
                normalization method: 'minmax' for :math:`(x - min(x))/(max(x) - min(x))`, 'standard' for :math:`(x - mean(x))/std(x)`, or 'none'

        Returns:
        --------
            df_train : pd.DataFrame
                training dataset with processed covariates
            df_test : pd.DataFrame
                testing dataset with processed covariates
        """

        if covariates is None: # if covariates are not provided, use all columns except duration_col and event_col
            covariates = df_train.columns.drop([duration_col, event_col])

        # check for cases where there is just one value
        for col in covariates:
            if len(df_train[col].unique()) == 1:
                raise ValueError(f"Column {col} has only one unique value. Please remove it from covariates.")

        # register metadata
        self.duration_col, self.event_col, self.covariates = duration_col, event_col, covariates

        X = pd.concat([df_train[covariates], df_test[covariates]])

        # find categorical covariates (type is 'category', or has less than 5 unique values)
        if categorical_covariates == True:
            categorical_covariates = []
            for col in covariates:
                if len(X[col].unique()) < 5:
                    categorical_covariates.append(col)
                elif X[col].dtype.name == 'category':
                    categorical_covariates.append(col)

        # encode categorical covariates via label-encoding
        if categorical_covariates:
            category_maps = {}
            for cat in categorical_covariates:
                category_maps[cat] = dict(enumerate(X[cat].astype('category').cat.categories))
                X[cat] = X[cat].astype('category').cat.codes
                X[cat] = X[cat].astype('float32')

        df_train[covariates] = X[:len(df_train)]
        df_test[covariates] = X[len(df_train):]

        if normalization is None or normalization == 'none':
            return df_train, df_test
        
        # detect high collinearity
        corr = pd.concat([df_train, df_test]).corr()
        np.fill_diagonal(corr.values, 0)
        if (np.abs(corr) > 0.999999).sum().sum() > 0:
            print("Warning: High collinearity detected. Consider removing one of the highly correlated features.")

        # normalize covariates
        normalizer = []
        if normalization == 'minmax':
            normalizer.append(X.min())
            normalizer.append(X.max() - X.min())
        elif normalization == 'standard':
            normalizer.append(X.mean())
            normalizer.append(X.std())
        else:
            raise NotImplementedError("Normalization can be 'minmax', 'standard' or 'none'.")

        df_train[covariates] = (df_train[covariates] - normalizer[0]) / normalizer[1]
        df_test[covariates] = (df_test[covariates] - normalizer[0]) / normalizer[1]

        # convert the keys of each of the category maps to be their normalized values
        if categorical_covariates:
            for cat in category_maps.keys():
                items = list(category_maps[cat].items())
                for key, val in items:
                    # remove the old key
                    category_maps[cat].pop(key)
                    # add the new key
                    new_key = round((key - normalizer[0][cat]) / normalizer[1][cat], 3)
                    category_maps[cat][new_key] = val
        
            # register
            self.categorical_covariates = categorical_covariates
            self.category_maps = category_maps

        # register normalizer
        self.normalizer = normalizer

        return df_train, df_test

    def train(self, df_train, df_val=None, do_prune_search=True, do_symbolic_fit=True, duration_col='duration', event_col='event', covariates=None,
              opt="Adam", lr=0.01, steps=100, batch=-1, early_stopping=False, stop_on='cindex',
              log=1, lamb=0., lamb_l1=1., lamb_entropy=0., 
              lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, stop_grid_update_step=50, 
              small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, 
              in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu', progress_bar=True,
              do_reg=True,prune_split = 0.25,prune_max_threshold=0.05, prune_bins=21, verbose=False):
        """
        Train the model.

        Args:
        -----
            df_train : pd.DataFrame
                training dataset
            df_val : pd.DataFrame
                validation dataset
            duration_col : str
                column name for duration
            event_col : str
                column name for event
            covariates : list
                list of covariates. If None, all columns except duration_col and event_col are used.
            opt : str
                optimizer. 'Adam' or 'LBFGS'
            lr : float
                learning rate
            steps : int
                number of steps
            batch : int
                batch size. If -1, use all samples.
            log : int
                log frequency
            lamb : float
                overall regularization strength
            lamb_l1 : float
                l1 regularization strength
            lamb_entropy : float
                entropy regularization strength
            lamb_coef : float
                spline coefficient regularization strength
            lamb_coefdiff : float
                spline coefficient difference regularization strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large factos
            metrics : list
                additional metrics to log
            sglr_avoid : bool
                avoid nan in SGLR
            save_fig : bool
                save figures
            beta : float
                beta for plotting
            save_fig_freq : int
                save figure frequency
            img_folder : str
                folder to save figures
            device : str
                device to use (no need to change as gpu is typically slower)

        Returns:
        --------
            log : dict
                log['train_loss'], 1D array of training losses (Cox loss)
                log['val_loss'], 1D array of val losses (Cox loss)
                log['train_cindex'], 1D array of training concordance index
                log['val_cindex'], 1D array of val concordance index
                log['reg'], 1D array of regularization (regularization in the total loss)
        """

        if do_prune_search:
            train,prune_opt = train_test_split(df_train, test_size=prune_split, random_state=42)
        else:
            train = df_train
            prune_opt = None
        
        # spline grid update frequency
        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        ### Register metadata
        if covariates is None: # if covariates are not provided, use all columns except duration_col and event_col
            covariates = train.columns.drop([duration_col, event_col])

        self.duration_col, self.event_col, self.covariates = duration_col, event_col, covariates

        ### Prepare data
        X_train = torch.tensor(train[covariates].values, dtype=torch.float32)
        y_train = torch.tensor(train[[duration_col, event_col]].values, dtype=torch.float32)
        if prune_opt is not None:
            X_val = torch.tensor(prune_opt[covariates].values, dtype=torch.float32)
            y_val = torch.tensor(prune_opt[[duration_col, event_col]].values, dtype=torch.float32)

        ### Define regularization
        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec+1e-7)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_
        
        ### Define optimizer
        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        ### Init log
        logger = Logger(early_stopping=early_stopping, stop_on=stop_on)
        logger['train_loss'], logger['val_loss'], logger['train_cindex'], logger['val_cindex'], logger['reg'] = [], [], [], [], []
        if metrics != None:
            for i in range(len(metrics)):
                logger[metrics[i].__name__] = []

        ### Define batch size
        if batch == -1 or batch > X_train.shape[0]:
            batch_size = X_train.shape[0]
        else:
            batch_size = batch

        ### Define closure (inner function for optimizer.step)
        global train_loss, reg_
        def closure(do_reg=do_reg):
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(X_train[train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = FastCoxLoss(pred[id_], y_train[train_id][id_].to(device))
            else:
                train_loss = FastCoxLoss(pred, y_train[train_id].to(device))
            if do_reg:
                reg_ = reg(self.acts_scale)
                loss = train_loss + lamb * reg_
            else:
                loss = train_loss
            loss.backward()
            return loss
        
        ### Generate best model hash for early stopping
        if early_stopping:
            best_model_hash = uuid.uuid4().hex
        
        if save_fig:
            os.makedirs(img_folder, exist_ok=True)

        ### Train
        if progress_bar: pbar = tqdm(range(steps), desc='description', ncols=100)
        else: pbar = range(steps)
        best_cindex = 0
        best_val_loss = np.inf
        for step, _ in enumerate(pbar):

            # Sample batch (typically, we use all samples for training)
            train_id = np.random.choice(X_train.shape[0], batch_size, replace=False)

            # Update spline grids
            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(X_train[train_id].to(device))

            # Update
            optimizer.step(closure)

            if metrics != None:
                for i in range(len(metrics)):
                    log[metrics[i].__name__].append(metrics[i]().item())

            logger['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            logger['train_cindex'].append(self.cindex(train))
            if do_reg: logger['reg'].append(reg_.cpu().detach().numpy())

            val_loss = FastCoxLoss(self.forward(X_val.to(device)), y_val.to(device))
            val_loss = torch.sqrt(val_loss).cpu().detach().numpy()
            cindex_val = self.cindex(prune_opt)
            if early_stopping and step > 1:
                if stop_on == 'cindex' and cindex_val > best_cindex:
                    best_cindex = cindex_val
                    self.save_ckpt(TEMP_CKPT_DIR / f'{best_model_hash}.pt', verbose=False)
                elif stop_on == 'loss' and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_ckpt(TEMP_CKPT_DIR / f'{best_model_hash}.pt', verbose=False)
            logger['val_loss'].append(val_loss)
            logger['val_cindex'].append(cindex_val)

            if _ % log == 0:
                if prune_opt is not None: pbar_desc = f"train loss: {logger['train_loss'][-1]:.2e} | val loss: {logger['val_loss'][-1]:.2e}"
                else: pbar_desc = f"train loss: {logger['train_loss'][-1]:.2e}"
                if progress_bar: pbar.set_description(pbar_desc)

            if save_fig and _ % save_fig_freq == 0:
                if in_vars is None: in_vars = list(covariates)
                if out_vars is None: out_vars = [r'$\hat{\theta}$']
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

        if early_stopping:
            self.load_ckpt(TEMP_CKPT_DIR / f'{best_model_hash}.pt', verbose=False)
            print('Best model loaded (early stopping).')
            os.remove(TEMP_CKPT_DIR / f'{best_model_hash}.pt')
            _ = self.predict(prune_opt) # necessary forward pass

        print(f'Training finished. Final Training C-index: {self.cindex(train):.3f}')
        if prune_opt is not None: print(f'Pre-pruned Validation C-index: {self.cindex(df_val):.3f}')
        else: print('No validation set provided; cannot compute C-index.')

        if do_prune_search:
            self.prune_edges(prune_opt, threshold_method='auto', verbose=verbose, max_search=prune_max_threshold, bins=prune_bins)
            print(f'Pruned Validation C-index: {self.cindex(df_val):.3f}')
            self.pruned = True

        if do_symbolic_fit:
            self.auto_symbolic(verbose=verbose)
            print(f'Symbolic Validation C-index: {self.cindex(df_val):.3f}')
            self.symbolic = True

        self.adjust_biases()

        return logger

    def cindex(self, df, duration_col=None, event_col=None):
        """
        Compute model's concordance index on a dataset.

        Args:
        -----
            df : pd.DataFrame
                dataset
            duration_col : str
                column name for duration
            event_col : str
                column name for event

        Returns:
        --------
            cindex : float
                concordance index
        """

        # if duration_col and event_col are not provided, use the registered metadata
        if duration_col is None and event_col is None:
            assert hasattr(self, 'duration_col') and hasattr(self, 'event_col'), "Dataset metadata not registered. Please train model or use process_data."
            duration_col, event_col = self.duration_col, self.event_col

        # compute concordance index
        X = torch.tensor(df.drop([self.duration_col, self.event_col], axis=1).values, dtype=torch.float32)
        log_ph = self(X).detach().numpy().flatten()

        return concordance_index(df[self.duration_col], -log_ph, df[self.event_col])

    def predict(self, df):
        """
        Predict log-partial hazard for all samples in a dataset.

        Args:
        -----
            df : pd.DataFrame
                dataset

        Returns:
        --------
            log_ph : pd.Series
                log-partial hazard
        """
    
        assert hasattr(self, 'duration_col') and hasattr(self, 'event_col') and hasattr(self, 'covariates'), "Dataset metadata not registered. Please train model or use process_data."
        X = torch.tensor(df[self.covariates].values, dtype=torch.float32)
        return pd.Series(self(X).cpu().detach().numpy().flatten(), index=df.index)

    def predict_partial_hazard(self, df):
        """
        Predict partial hazard for all samples in a dataset (exp of log-partial hazard).

        Args:
        -----
            df : pd.DataFrame
                dataset

        Returns:
        --------
            partial_hazard : pd.Series
                partial hazard
        """
        return np.exp(self.predict(df))

    def find_prune_threshold(self, val_df, max=0.05, bins=21,cache_loc=os.getcwd(),
                             cache_store_name='__modelcache__.pt',verbose=True):

        cache_fp = cache_loc + '/' + cache_store_name
        self.save_ckpt(cache_fp, verbose=False)
        pruning_thresholds = np.linspace(0, max, bins)
        cindices = np.zeros(len(pruning_thresholds))

        for i, threshold in enumerate(pruning_thresholds):
            ckan_ = CoxKAN(**self.config)
            ckan_.load_ckpt(cache_fp, verbose=False)
            _ = ckan_.predict(val_df)  # important forward pass after loading a model

            prunable = True
            for l in range(ckan_.depth):
                if not (ckan_.acts_scale[l] > threshold).any():
                    prunable = False
                    break

            ckan_ = ckan_.prune_nodes(threshold)
            if 0 in ckan_.width: prunable = False
            if not prunable:break

            _ = ckan_.predict(val_df)  # important forward pass
            ckan_.prune_edges(val_df, threshold_method='set',threshold=threshold,verbose=False)

            cindices[i] = ckan_.cindex(val_df)
            if verbose: print(f'Pruning threshold: {threshold:.4f}, C-Index (Val): {cindices[i]:.4f}')
        max_cindex = np.max(cindices)
        best_threshold = np.max(pruning_thresholds[cindices == max_cindex])
        if np.max(cindices) < 0.51: best_threshold = 0
        #delete model cache
        os.remove(cache_fp)

        return best_threshold

    
    def prune_edges(self, df, threshold_method ='auto',threshold=0.02,max_search=0.05,bins=21, verbose=True):
        """
        Prune edges (activation functions) of the model based on a threshold of the L1 norm
        of that activation.

        Args:
        -----
            threshold : float
                any activation with L1 norm less than this threshold will be pruned
            verbose : bool
                If True, print pruned activations

        Returns:
        --------
            None
        """

        #find threshold
        if threshold_method == 'auto':
            threshold = self.find_prune_threshold(df,max=max_search, bins=bins, verbose=verbose)
        elif threshold_method == 'set':
            #do nothing
            pass
        else:
            raise NotImplementedError("threshold_method can be 'auto' or 'set'.")

        if verbose: print(f'Pruning threshold: {threshold:.3f}')

        # loop through all activations
        for l in range(self.depth):
            for i in range(self.width[l]):
                for j in range(self.width[l+1]):
                    if self.acts_scale[l][j][i] < threshold:
                        super(CoxKAN, self).remove_edge(l, i, j)        # remove edge
                        self.fix_symbolic(l, i, j, '0', verbose=False)  # set symbolic activation to 0
                        self.acts_scale[l][j][i] = 0                    # set scale to 0

                        if verbose: print(f'Pruned activation ({l},{i},{j})')
                        assert self.symbolic_fun[l].funs_name[j][i] == '0'
                        assert self.symbolic_fun[l].mask[j][i] == 1
                        assert self.act_fun[l].mask[j * self.width[l] + i] == 0
                        assert self.acts_scale[l][j][i] == 0

    def prune_nodes(self, threshold=1e-2, mode="auto", active_neurons_id=None):
        '''
        Prune nodes (neurons) of the model based on a threshold of the L1 norm of the incoming
        and outgoing activations of that neuron. This method is just slightly adapted from
        the original KAN.prune().
        
        Args:
        -----
            threshold : float
                any neuron which has all incoming and outgoing activations with L1 norm less than this threshold will be pruned
            mode : str
                "auto" or "manual". If "auto", the thresold will be used to automatically prune away nodes. If "manual", active_neuron_id is needed to specify which neurons are kept (others are thrown away).
            active_neuron_id : list of id lists
                For example, [[0,1],[0,2,3]] means keeping the 0/1 neuron in the 1st hidden layer and the 0/2/3 neuron in the 2nd hidden layer. Pruning input and output neurons is not supported yet.
            
        Returns:
        --------
            model2 : CoxKAN
                pruned model
        '''
        mask = [torch.ones(self.width[0], )]
        active_neurons = [list(range(self.width[0]))]
        for i in range(len(self.acts_scale) - 1):
            if mode == "auto":
                in_important = torch.max(self.acts_scale[i], dim=1)[0] > threshold
                out_important = torch.max(self.acts_scale[i + 1], dim=0)[0] > threshold
                overall_important = in_important * out_important
            elif mode == "manual":
                overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                overall_important[active_neurons_id[i + 1]] = True
            mask.append(overall_important.float())
            active_neurons.append(torch.where(overall_important == True)[0])
        active_neurons.append(list(range(self.width[-1])))
        mask.append(torch.ones(self.width[-1], ))

        self.mask = mask  # this is neuron mask for the whole model

        # update act_fun[l].mask
        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width[l + 1]):
                if i not in active_neurons[l + 1]:
                    self.remove_node(l + 1, i)

        model2 = CoxKAN(width=copy.deepcopy(self.width), grid=self.grid, k=self.k, base_fun=self.base_fun, device='cpu')
        model2.load_state_dict(self.state_dict(), strict=False)

        # copy other attributes
        dic = {}
        for k, v in self.__dict__.items():
            if k[0] != '_':
                setattr(model2, k, v)

        for i in range(len(self.acts_scale)):
            if i < len(self.acts_scale) - 1:
                model2.biases[i].weight.data = model2.biases[i].weight.data[:, active_neurons[i + 1]]

            model2.act_fun[i] = model2.act_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])
            model2.width[i] = len(active_neurons[i])
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])

        #set model2 biases in final layer to 0
        model2.biases[-1].weight.data = torch.zeros(model2.width[-1], dtype=torch.float32)

        return model2
    
    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True, random=False):
        '''
        Set (l,i,j) activation to be symbolic (specified by fun_name).
        
        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            fun_name : str
                function name
            fit_params_bool : bool
                obtaining affine parameters through fitting (True) or setting default values (False)
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : bool
                If True, more information is printed.
            random : bool
                initialize affine parameteres randomly or as [1,0,1,0]
        
        Returns:
        --------
            None or r2 (coefficient of determination)
  
        '''
        self.set_mode(l, i, j, mode="s")
        if fun_name == 'categorical':
            assert l == 0, "Only input layer can have categorical activations"
            x = self.acts[l][:, i]
            y = self.spline_postacts[l][:, j, i]
            category_map = self.category_maps[self.covariates[i]]
            fun, fun_sympy = categorical_fun(inputs=x, outputs=y, category_map=category_map)
            self.symbolic_fun[l].funs_sympy[j][i] = fun_sympy
            self.symbolic_fun[l].funs_name[j][i] = fun_name
            self.symbolic_fun[l].funs[j][i] = fun 
            self.symbolic_fun[l].affine.data[j][i] = torch.tensor([1.,0.,1.,0.])
            return None
        elif not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random)
            return None
        else:
            x = self.acts[l][:, i]
            y = self.spline_postacts[l][:, j, i]
            r2 = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, x, y, a_range=a_range, b_range=b_range, verbose=verbose)

            # if in output layer, fix output bias to zero
            if l == self.depth - 1:
                self.symbolic_fun[l].affine.data[j][i][3] = 0.

            return r2

    def plot(self, show_vars=False, **kwargs):
        """ Plot the model. 
        
        Args:
        -----
            show_vars : bool
                If True, show the registered covariates on the plot. Default: False
            **kwargs : Keyword arguments to be passed to KAN.plot()
        
        Keyword Args:   
        -------------
            folder : str
                the folder to store pngs
            beta : float
                positive number. control the transparency of each activation. transparency = tanh(beta*l1).
            mask : bool
                If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
            mode : bool
                "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
            scale : float
                control the size of the diagram
            in_vars: None or list of str
                the name(s) of input variables
            out_vars: None or list of str
                the name(s) of output variables
            title: None or str
                title

        Returns:
        --------
            fig : Figure
                the figure
        """

        # re-apply mask
        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    if  self.symbolic_fun[l].funs_name[j][i] == '0' and self.symbolic_fun[l].mask[j, i] > 0.:
                        self.acts_scale[l][j][i] = 0.

        if show_vars:
            super(CoxKAN, self).plot(in_vars=list(self.covariates), out_vars=[r'$\hat{\theta}(\mathbf{x})$'], **kwargs)
        else:
            super(CoxKAN, self).plot(**kwargs)
        return plt.gcf()

    def plot_act(self, l, i, j):
        """
        Plot activation function phi_(l,i,j)

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
        """
        # obtain inputs (pre-activations) and outputs (post-activations)
        inputs = self.spline_preacts[l][:,j,i]
        outputs = self.spline_postacts[l][:,j,i]

        # they are not ordered yet
        rank = np.argsort(inputs)
        inputs = inputs[rank]
        outputs = outputs[rank]

        fig = plt.figure()
        plt.plot(inputs, outputs, marker="o")
        return fig

    def suggest_symbolic(self, l, i, j, a_range=(-10, 10), b_range=(-10, 10), lib=None, topk=5, verbose=True,
                         high_corr=0.9):
        '''
        Suggest the symbolic candidates of activation function phi_(l,i,j)

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            lib : dic
                library of symbolic bases. If lib = None, the global default library will be used.
            topk : int
                display the top k symbolic functions (according to r2)
            verbose : bool
                If True, more information will be printed.

        Returns:
        --------
            fun_name : str
                suggested symbolic function name
            fun : fun
                suggested symbolic function
            r2 : float
                coefficient of determination of best suggestion

        '''

        # Shortcut: If the network has categorical covariates and we're at layer 0,
        # and if the covariate is categorical while the pre-fitted function is not '0',
        # then return immediately a categorical suggestion.
        if hasattr(self, 'categorical_covariates') and l == 0:
            if self.covariates[i] in self.categorical_covariates and self.symbolic_fun[l].funs_name[j][i] != '0':
                return 'categorical', None, 1

        # Initialize lists to store r2 scores and attempted symbolic functions.
        r2_scores = []
        attempted_sym_lib = []

        # Choose the appropriate symbolic library.
        if lib is None:
            symbolic_lib = SYMBOLIC_LIB
        else:
            # Use only the entries in the provided lib.
            symbolic_lib = {key: SYMBOLIC_LIB[key] for key in lib}

        # remove entries from symbolic_lib that are not in ordered_functions
        symbolic_lib = {key: symbolic_lib[key] for key in ordered_functions if key in symbolic_lib}

        if verbose:
            print(f"Suggesting symbolic function for layer {l}, input index {i}, output index {j}")

        # Try fitting each candidate symbolic function.
        for name, fn in symbolic_lib.items():
            try:
                r2 = self.fix_symbolic(l, i, j, name, a_range=a_range, b_range=b_range, verbose=False)
                if verbose:
                    print(name, r2, fn)
                r2_scores.append(r2.item())
                attempted_sym_lib.append((name, fn))
            except Exception as err:
                if verbose:
                    print(f'Error in fitting "{name}": {err}')

        # Revert any temporary changes made by fix_symbolic.
        self.unfix_symbolic(l, i, j)

        # Sort the attempted functions based on their predefined order in ordered_functions.
        sorted_indices = np.argsort([ordered_functions.index(item[0]) for item in attempted_sym_lib])
        r2_scores = np.array(r2_scores)[sorted_indices]

        # we can use the softmax of the r2 scores to determine the best candidate
        if len(sorted_indices) > 0:
            # first, normally distribute the r2 scores - account for edge cases to prevent nans and infs
            norm_r2_scores = np.clip(r2_scores, 1e-7, 1 - 1e-7)
            norm_r2_scores = (norm_r2_scores - np.mean(norm_r2_scores)) / (np.std(norm_r2_scores)+1e-7)
            softmax_r2 = np.exp(norm_r2_scores) / np.sum(np.exp(norm_r2_scores)+1e-7)

            # if the max of softmax is greater than 0.5, return the best candidate
            if np.max(softmax_r2) > 0.5:
                best_index = np.argmax(softmax_r2)
                if verbose:
                    print("Returning the best candidate based on softmax v1.")
                return attempted_sym_lib[sorted_indices[best_index]][0], attempted_sym_lib[sorted_indices[best_index]][1], r2_scores[best_index]

            # otherwise, check if max of softmax is greater than 0.2 higher than second max - only viable if three or more functions contending.
            elif len(softmax_r2) > 2 and (np.max(softmax_r2) - np.partition(softmax_r2, -2)[-2]) > 0.2:
                best_index = np.argmax(softmax_r2)
                if verbose:
                    print("Returning the best candidate based on softmax v2.")
                return attempted_sym_lib[sorted_indices[best_index]][0], attempted_sym_lib[sorted_indices[best_index]][
                    1], r2_scores[best_index]

        # if no r2 exceeds high_corr, return highest r2 func
        if max(r2_scores) < high_corr:
            if verbose:
                print("No symbolic function with r2 > high_corr found. Returning the best of bad candidates.")
            best_index = np.argmax(r2_scores)
            return attempted_sym_lib[sorted_indices[best_index]][0], attempted_sym_lib[sorted_indices[best_index]][1], r2_scores[best_index]

        # Filter out candidates with r2 less than high_corr and select the simplest remaining candidates.
        valid_mask = r2_scores >= high_corr
        sorted_indices = sorted_indices[valid_mask]
        r2_scores = r2_scores[valid_mask]
        attempted_sym_lib = [attempted_sym_lib[i] for i in sorted_indices]

        # Re-sort for the remaining valid candidates.
        sorted_indices = np.argsort([ordered_functions.index(item[0]) for item in attempted_sym_lib])
        r2_scores = np.array(r2_scores)[sorted_indices]

        if verbose:
            print("Final attempted library:", attempted_sym_lib)
            print("Final sorted indices:", sorted_indices)
            if len(sorted_indices) > 0:
                print("Best candidate index:", sorted_indices[0])
                print("Best candidate:", attempted_sym_lib[sorted_indices[0]])

        best_name = attempted_sym_lib[sorted_indices[0]][0]
        best_fn = attempted_sym_lib[sorted_indices[0]][1]
        best_r2 = r2_scores[0]

        return best_name, best_fn, best_r2

    def plot_best_suggestion(l, i, j, lib=None, a_range=(-10,10), b_range=(-10,10), verbose=1):
        """
        Plot the best symbolic suggestion for activation function phi_(l,i,j)

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            lib : None or a list of function names
                the symbolic library 
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : int
                verbosity

        Returns:
        --------
            fig : Figure
                the figure
        """
        x = self.spline_preacts[l][:,j,i]
        y = self.spline_postacts[l][:,j,i]

        # they are not ordered yet
        rank = np.argsort(x)
        x = x[rank]
        y = y[rank]

        fn_name, _, r2 = self.suggest_symbolic(l, i, j, lib=lib, a_range=a_range, b_range=b_range, verbose=verbose)

        # minimise |y-(cf(ax+b)+d)|^2 w.r.t a,b,c,d
        func = SYMBOLIC_LIB[fn_name][0]
        (a, b, c, d), r2 = fit_params(x, y, func, a_range=a_range, b_range=b_range, verbose=verbose)

        y_pred = c*func(a*x+b)+d

        fig, ax = plt.subplots()

        ax.scatter(x, y, label="Activation")
        ax.plot(x, y_pred, color='red', linestyle='--', label=f"Symbolic Fit")
        ax.set_title(f"{c:.3f}{fn_name}({a:.3f}x + {b:.3f}) + {d:.3f}")
        ax.legend()
        return fig

    def auto_symbolic(self, min_r2=0,high_corr=0.9, a_range=(-10, 10), b_range=(-10, 10), lib=None, verbose=1):
        '''
        Automatic symbolic regression: using best suggestion from suggest_symbolic to replace activations with symbolic functions.
        This method is just slightly adapted from the original KAN.auto_symbolic().
        
        Args:
        -----
            min_r2 : float
                minimum r2 to accept the symbolic formula
            lib : None or a list of function names
                the symbolic library 
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : int
                verbosity
                
        Returns:
        --------
            bool: True if all activations are successfully replaced by symbolic functions, False otherwise
        '''

        for l in range(self.depth):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    if self.symbolic_fun[l].mask[j, i] > 0.:
                        if verbose:
                            print(f'skipping ({l},{i},{j}) since already symbolic')
                    else:
                        name, fn, r2 = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib, verbose=verbose,
                                                             high_corr=high_corr)
                        if r2 >= min_r2:
                            self.fix_symbolic(l, i, j, name, verbose=verbose > 1)
                            if verbose >= 1:
                                print(f'fixing ({l},{i},{j}) with {name}, r2={r2}')
                                print(f'({l},{i},{j})', name, r2)
                        else:
                            print(f'No symbolic formula found for ({l},{i},{j})')
                            return False



        return True


    def symbolic_formula(self, floating_digit=None, var=None, normalizer=None, simplify=False, output_normalizer = None ):
        '''
        Obtain the symbolic formula.
        
        Args:
        -----
            floating_digit : int
                the number of digits to display
            var : list of str
                the name of variables (if not provided, by default using ['x_1', 'x_2', ...])
            normalizer : [mean array (floats), varaince array (floats)]
                the normalization applied to inputs
            simplify : bool
                If True, simplify the equation at each step (usually quite slow), so set up False by default.
            output_normalizer: [mean array (floats), varaince array (floats)]
                the normalization applied to outputs
            
        Returns:
        --------
            symbolic formula : sympy function
                the symbolic formula
            x0 : list of sympy symbols
                the list of input variables
        
        '''
        symbolic_acts = []
        x = []

        def ex_round(ex1, floating_digit=floating_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, floating_digit))
            return ex2
        
        if normalizer is None and hasattr(self, 'normalizer'):
            normalizer = self.normalizer

        # define variables
        if var is None:
            if hasattr(self, 'covariates'):
                x = [sympy.symbols(var_.replace(' ', '_')) for var_ in self.covariates]
            else:
                for ii in range(1, self.width[0] + 1):
                    exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                    exec(f"x.append(x{ii})")
        else:
            x = [sympy.symbols(var_.replace(' ', '_')) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = np.array(normalizer[0])
            std = np.array(normalizer[1])
            if hasattr(self, 'categorical_covariates'):
                for i, var_ in enumerate(self.covariates):
                    if var_ not in self.categorical_covariates:
                        x[i] = (x[i] - mean[i]) / std[i]
            else:
                x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        self.adjust_biases()

        for l in range(self.depth):
            y = []
            for j in range(self.width[l + 1]):
                yj = 0.
                for i in range(self.width[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    fun_name = self.symbolic_fun[l].funs_name[j][i]

                    # print(sympy_fun, fun_name, a, b, c, d)
                    try:
                        if fun_name == 'categorical':
                            assert a == 1 and b == 0 and c == 1 and d == 0
                            yj += sympy_fun(x[i])
                        else:
                            yj += c * sympy_fun(a * x[i] + b) + d
                    except Exception as e:
                        print('Error: ', e)

                if simplify == True:
                    y.append(sympy.simplify(yj))
                else:
                    y.append(yj)


            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
            assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'
            
            output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
            symbolic_acts[-1] = output_layer


        if floating_digit is None:
            return symbolic_acts[-1], x0

        self.symbolic_acts = [[ex_round(symbolic_acts[l][i]) for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]

        out_dim = len(symbolic_acts[-1])
        return [ex_round(symbolic_acts[-1][i]) for i in range(len(symbolic_acts[-1]))], x0

    def save_ckpt(self, save_path='ckpt.pt', verbose=True):
        ''' Save the current model as checkpoint '''

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'state_dict': self.state_dict(),
        }
        for k, v in self.__dict__.items():
            if k[0] != '_':
                state[k] = v

        torch.save(state, save_path)
        if verbose: print(f'Saved model to {save_path}')

    def load_ckpt(self, ckpt_path, verbose=True):
        ''' Load model from checkpoint '''
        if verbose: print(f'Loading model from {ckpt_path}...')
        state = torch.load(ckpt_path,weights_only=False)
        self.load_state_dict(state['state_dict'],strict=False)
        for k, v in state.items():
            if k != 'state_dict':
                setattr(self, k, v)
        if verbose: print(f'Loaded model from {ckpt_path}')

    def symbolic_rank_terms(self, floating_digit=5, z_score_threshold=5, normalizer=None):
        """
        Calculate the standard deviation of each term in the symbolic expression of CoxKAN.

        Standard deviation can be used as a measure of importance of each term in the symbolic expression.
        The terms with higher standard deviation are more important. A caveat here is that terms with
        outliers in their outputs may have higher standard deviation, which may not necessarily mean they
        are more important. To address this, we remove outliers iteratively based on Z-score until no
        outliers are left.

        Args:
        -----
            floating_digit : int
                the number of digits to display
            z_score_threshold : int
                the threshold of Z-score for removing outliers
            normalizer : [mean array (floats), varaince array (floats)]
                the normalization applied to inputs

        Returns:
        --------
            terms_std : dict
                dictionary of terms and their standard deviations
        """

        for l in range(self.depth):
            if not (self.symbolic_fun[l].mask == 1).all():
                raise ValueError('All activation functions must be symbolic for ranking.')
            
        def zscore(arr):
            return (arr - np.mean(arr)) / np.std(arr)
            
        def remove_outliers(arr):
            """ 
            Remove outliers from an array based on Z-score iteratively until no outliers are left.
            """
            z_scores = np.abs(zscore(arr))
            while np.any(z_scores > z_score_threshold):
                arr = arr[np.abs(zscore(arr)) < z_score_threshold]
                z_scores = np.abs(zscore(arr))
            return arr

        def ex_round(ex1):
            """
            Round the floating point numbers in a sympy expression.
            """
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, floating_digit))
            return ex2

        if normalizer is None and hasattr(self, 'normalizer'):
            normalizer = self.normalizer

        with torch.no_grad():
            ### Get the terms in the KAN symbolic expression and their standard deviations
            terms_std = {}

            if len(self.width) == 2:
                for i in range(self.width[0]):
                    l = 0; j = 0
                    fun_name = self.symbolic_fun[l].funs_name[j][i]
                    fun = self.symbolic_fun[l].funs[j][i]
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]

                    if fun_name != '0':
                        outputs = self.spline_postacts[l][:, j, i].detach().numpy().flatten()
                        outputs = remove_outliers(outputs)
                        x = self.covariates[i]
                        x = x.replace(' ', '_')
                        if normalizer is not None:
                            x = (sympy.symbols(x) - normalizer[0][i]) / normalizer[1][i]
                        else: 
                            x = sympy.symbols(x)
                        sympy_trans = self.symbolic_fun[l].funs_sympy[j][i]
                        term = c * sympy_trans(a * x + b)
                        term = ex_round(term)
                        term = f'({l},{i},{j}) ' + str(term)
                        terms_std[term] = outputs.std()
                        
            elif len(self.width) == 3:
                for i in range(self.width[1]):
                    l = 1
                    j = 0
                    fun_name = self.symbolic_fun[l].funs_name[j][i]
                    fun = self.symbolic_fun[l].funs[j][i]
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    if fun_name != '0':
                        # if the final layer activation is non-linear, it is an interaction term
                        if fun_name != 'x':
                            outputs = self.spline_postacts[l][:, j, i].detach().numpy().flatten()
                            outputs = remove_outliers(outputs)
                            fun_name = f'({l},{i},{j}) {fun_name} interaction term'
                            terms_std[fun_name] = outputs.std()
                        # if the final layer activation is linear, it contains many isolation terms
                        else:
                            assert fun(1) == 1, f'Function {fun} is not linear'
                            j = i # current node becomes the output node
                            l = 0 # input layer
                            for i in range(self.width[l]):
                                if self.symbolic_fun[l].funs_name[j][i] != '0':

                                    # calculate standard deviation of the term (excluding outliers)
                                    outputs = self.spline_postacts[l][:, j, i].detach().numpy().flatten()
                                    outputs = c * (a * outputs + b) + d
                                    outputs = remove_outliers(outputs)
                                    std = outputs.std().item()

                                    # get symbolic expression of the term
                                    x = self.covariates[i]
                                    x = x.replace(' ', '_')
                                    if normalizer is not None:
                                        x = (sympy.symbols(x) - normalizer[0][i]) / normalizer[1][i]
                                    else: 
                                        x = sympy.symbols(x)
                                    a_, b_, c_, d_ = self.symbolic_fun[l].affine[j, i]
                                    sympy_trans = self.symbolic_fun[l].funs_sympy[j][i]
                                    transformation = c_ * sympy_trans(a_ * x + b_)
                                    term = c * a * transformation
                                    term = ex_round(term)
                                    term = f'({l},{i},{j}) ' + str(term)

                                    terms_std[term] = std
            else:
                raise NotImplementedError('Ranking terms is currently only supported for models with up to 1 hidden layer.')
        return terms_std

    def adjust_biases(self):
        """
        Recursively adjusts biases (translational affine biases used to fit symbolic functions)
        in the network along every branch that forms a linear connection from the output back toward the input.

        For each affine mapping at layer l (mapping neurons in layer l to layer l+1):
          - The external bias 'd' is removed (set to zero) for every connection that
            eventually contributes to the output.
          - If a connection uses a linear activation (activation function is 'x'),
            the internal bias 'b' is removed, and the input neuron (index in layer l)
            is marked for further backpropagation.

        Indexing convention:
          - l: layer index (an affine mapping from layer l to layer l+1)
          - j: index of the input neuron (in layer l)
          - i: index of the output neuron (in layer l+1)

        Assumptions:
          - self.depth: the total number of affine layers.
          - self.width: a list with the number of neurons per layer (layer 0 is input, layer self.depth is output).
          - self.symbolic_fun[l].affine[j, i]: a torch.Tensor containing a tuple (a, b, c, d)
                for the connection from neuron j in layer l to neuron i in layer l+1.
          - self.symbolic_fun[l].funs_name[j][i]: a string holding the activation function name
                for that connection (with 'x' representing a linear activation).
        """
        LINEAR_ACTIVATION = 'x'

        with torch.no_grad():
            def recursive_adjust(l, upstream_set):
                """
                Recursively adjust layer l and propagate backwards.

                Args:
                  l: the current affine layer (mapping from layer l to layer l+1).
                  upstream_set: a set of indices (of the output side of layer l) that lie on a
                                linear chain connecting to the final output.

                For each connection (from neuron j in layer l to neuron i in layer l+1) where i is in
                upstream_set, we set the external bias (d) to zero unconditionally. If the activation
                for that connection is linear (funs_name equals 'x'), we also set the internal bias (b) to zero
                and add neuron j to a new upstream set for layer (l-1).
                """
                # Base case: nothing to adjust if no layer remains or if no neurons are marked upstream.
                if l < 0 or not upstream_set:
                    return

                out_dim = self.symbolic_fun[l].out_dim
                in_dim = self.symbolic_fun[l].in_dim
                new_upstream = set()  # will hold indices in layer l that connect linearly upward
                for j in range(out_dim):  # iterate over input neurons of layer l
                    if j in upstream_set:
                        for i in range(in_dim):  # iterate over output neurons of layer l+1
                            # Retrieve the parameters for connection from neuron j to neuron i.
                            a, b, c, d = self.symbolic_fun[l].affine[j, i]
                            # Remove external bias in every connection.
                            d = 0

                            if len(self.biases[l].weight.data.shape) == 1:
                                self.biases[l].weight.data[j] = 0
                            elif len(self.biases[l].weight.data.shape) == 2:
                                assert self.biases[l].weight.data.shape[0] == 1
                                self.biases[l].weight.data[0, j] = 0
                            else:
                                raise ValueError('Bias shape not supported: {}'.format(self.biases[l].weight.data.shape))

                            # If activation function is linear, remove internal bias and mark input neuron j.
                            if self.symbolic_fun[l].funs_name[j][i] == LINEAR_ACTIVATION:
                                b = 0
                                new_upstream.add(i)
                            # Update the tensor using an in-place copy to avoid gradient issues.
                            self.symbolic_fun[l].affine[j, i].copy_(torch.tensor([a, b, c, d], dtype=torch.double))

                # Recurse: adjust the previous layer (l-1) using the new upstream set.
                recursive_adjust(l - 1, new_upstream)

            # For the final affine layer (l = self.depth-1) mapping from layer (self.depth-1) to output (layer self.depth),
            # we start with all output neurons as the upstream set.
            output_width = self.symbolic_fun[-1].in_dim
            initial_upstream = set(range(output_width))
            recursive_adjust(self.depth - 1, initial_upstream)