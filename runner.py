import os
import platform
import re
import shutil
import sys
import time
from collections import OrderedDict
from typing import Optional, Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
import wandb
from torch.cuda.amp import autocast
from torch.utils.data import WeightedRandomSampler
from torchmetrics import MeanMetric
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from models.model_wrappers import RescaleHeadWrapper

import visualization
from datasetClass import SatelliteImageDataset
from metrics import MetricsClass
from utilities import SequentialSchedulers

from models.unet_3d_single_year import UNetSixMonth as SingleYearUNetSixMonth
from models.unet_3d_single_year import UNetTwelveMonth as SingleYearUNetTwelveMonth

class Runner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config: Any, tmp_dir: str, debug: bool):
        """
        Initialize useful variables using config.
        :param config: wandb run config
        :type config: wandb.config.Config
        :param debug: Whether we are in debug mode or not
        :type debug: bool
        """
        self.config = config
        self.debug = debug

        # Set the device
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"Number of GPUs available: {n_gpus}")
            config.update(dict(device='cuda:0'))
        else:
            print("No GPUs available.")
            config.update(dict(device='cpu'))

        self.dataParallel = (torch.cuda.device_count() > 1)
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all visible GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)
        torch.backends.cudnn.benchmark = True

        # Set a couple useful variables
        self.seed = int(self.config.seed)
        self.loss_name = self.config.loss_name or 'shift_l1'
        sys.stdout.write(f"Using loss: {self.loss_name}.\n")
        self.use_amp = self.config.fp16
        self.tmp_dir = tmp_dir
        print(f"Using temporary directory {self.tmp_dir}.")
        self.is_3d_model = '3d' in self.config.arch
        self.hardcoded_base_years = [2019, 2020, 2021, 2022]
        # Check if the dataset includes sentinel-1 data (indicated by the string 's1_s2')
        self.has_sentinel_1 = 's1_s2' in self.config.dataset
        self.requires_both_random_and_same = 'both' in self.config.dataset  # If True, we need to copy both random and same datasets and create a joint dataloader, taking validation only from the random dataset
        
        # Check if the dataset name specifies a specific year, e.g. ai4forest_2020_12_12_256_256 or ai4forest_same_s1_s2_2020_12_12_256_256
        for year in self.hardcoded_base_years:
            if f'{year}_' in self.config.dataset:
                self.hardcoded_base_years = [year]
                break
        self.shift_year = 0

        # Variables to be set
        self.loader = {loader_type: None for loader_type in ['train', 'val']}
        self.loss_criteria = {loss_name: self.get_loss(loss_name=loss_name) for loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber']}
        for threshold in [15, 20, 25, 30]:
            self.loss_criteria[f"l1_{threshold}"] = self.get_loss(loss_name=f"l1", threshold=threshold)

        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.artifact = None
        self.model_paths = {model_type: None for model_type in ['initial', 'trained']}
        self.model_metrics = {  # Organized way of saving metrics needed for retraining etc.
        }

        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'shift_l1': MeanMetric().to(device=self.device),
                               'shift_l2': MeanMetric().to(device=self.device),
                               'shift_huber': MeanMetric().to(device=self.device),
                               'l1': MeanMetric().to(device=self.device),
                               'l2': MeanMetric().to(device=self.device),
                               'huber': MeanMetric().to(device=self.device),
                               }
                        for mode in ['train', 'val']}

        for mode in ['train', 'val']:
            for threshold in [15, 20, 25, 30]:
                self.metrics[mode][f"l1_{threshold}"] = MeanMetric().to(device=self.device)
        
        # Add the metrics for each year
        for year in self.hardcoded_base_years:
            self.metrics[f'val_{year}'] = {metric_name: MeanMetric().to(device=self.device) for metric_name in self.metrics['val'].keys()}

        self.metrics['train']['ips_throughput'] = MeanMetric().to(device=self.device)

        self.use_early_stopping = config.early_stopping and (not self.debug)
        self.best_val_loss = float('inf')
        self.metrics['best_val'] = {metric_name: MeanMetric().to(device=self.device) for metric_name in self.metrics['val'].keys()}
        # Add the metrics for each year
        for year in self.hardcoded_base_years:
            self.metrics[f'best_val_{year}'] = {metric_name: MeanMetric().to(device=self.device) for metric_name in self.metrics['val'].keys()}
        self.best_model_path = None  # Path to save the best model

    @staticmethod
    def set_seed(seed: int):
        """
        Sets the seed for the current run.
        :param seed: seed to be used
        """
        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self) -> dict:
        """
        Returns the metrics for the current epoch.
        :return: dict containing the metrics
        :rtype: dict
        """
        with torch.no_grad():
            loggingDict = dict(
                # Model metrics
                n_params=MetricsClass.get_parameter_count(model=self.model),

                # Optimizer metrics
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
            )
            # Add metrics
            for split in ['train', 'val']:
                for metric_name, metric in self.metrics[split].items():
                    try:
                        # Catch case where MeanMetric mode not set yet
                        loggingDict[f"{split}/{metric_name}"] = metric.compute()
                    except Exception as e:
                        continue
            # Add the metrics for each year
            for year in self.hardcoded_base_years:
                for metric_name, metric in self.metrics[f'val_{year}'].items():
                    loggingDict[f'val_{year}/{metric_name}'] = metric.compute()

        return loggingDict

    @staticmethod
    def get_dataset_root(dataset_name: str) -> str: # TODO: this is not yet adapted to the new caching system
        """Returns the dataset rootpath for the given dataset.
        If we are on a z1 node and the dataset is not cached locally, it will be copied to the local cache."""
        sys.stdout.write(f"Loading {dataset_name}.\n")
        is_htc = 'htc-' in platform.uname().node
        is_coder = 'coder-' in platform.uname().node
        is_gcp = 'gcp-' in platform.uname().node
        is_palma = bool(re.match(r'r\d+n\d+', platform.uname().node))

        # Determine where the data lies
        for permanent_cache in ['/software/ais2t/pytorch_datasets', '/home/htc/mzimmer/SCRATCH', '/home/jovyan/work/scratch', '/scratch/tmp/j_paul17/datasets']:  # AIS2T Permanent Storage, AIS2T SCRATCH, scratch_jan, palma_jan
            permanent_dataset_root = os.path.join(permanent_cache, dataset_name)
            if os.path.isdir(permanent_dataset_root):
                break

        #local_cache = '/scratch/local/ais2t_vision_cache'
        local_cache = '/scratch/local'
        local_dataset_root = os.path.join(local_cache, dataset_name)

        dataset_is_permanently_cached = os.path.exists(permanent_dataset_root)
        assert is_htc or is_gcp or is_coder or is_palma, "This function is only intended to be used on htc or coder nodes and currently only allows already downloaded datasets."
        assert dataset_is_permanently_cached, "The dataset is not permanently cached on z1, aborting."
        if is_gcp:
            assert permanent_cache == '/home/htc/mzimmer/SCRATCH', "GCP needs datasets on GCP storage."

        is_copyable = dataset_name in ['ai4forest_6_12_256_256', 'ai4forest_same_s1_s2_6_12_256_256', 'ai4forest_random_s1_s2_6_12_256_256', 'ai4forest_2020_12_12_256_256']
        if is_htc and is_copyable:
            # We are running on a z1 node -> check whether the dataset is cached locally, otherwise cache it locally from the permanent cache
            busyFile = os.path.join(local_cache, f"{dataset_name}-busyfile.lock")
            doneFile = os.path.join(local_cache, f"{dataset_name}-donefile.lock")
            if os.path.exists(local_dataset_root) and os.path.exists(doneFile):
                # The dataset is cached locally on the node and we can use it
                dataset_root = local_dataset_root
            else:
                # The dataset is not cached locally, hence we need to copy it
                os.makedirs(local_cache, exist_ok=True) # Create the cache directory if it does not exist

                wait_it = 0
                while True:
                    is_done = os.path.exists(doneFile)
                    is_busy = os.path.exists(busyFile)
                    if is_done:
                        # Dataset exists locally
                        dataset_root = local_dataset_root
                        sys.stdout.write("Local data storage: Done file exists.\n")
                        break
                    elif is_busy:
                        # Wait for 10 seconds, then check again
                        time.sleep(10)
                        sys.stdout.write("Local data storage: Is still busy - wait.\n")
                        continue
                    else:
                        # Create the busyFile
                        open(busyFile, mode='a').close()

                        # Copy the dataset
                        sys.stdout.write(f"Local data storage: Starts copying from {permanent_dataset_root} to {local_dataset_root}.\n")
                        try:
                            shutil.copytree(src=permanent_dataset_root, dst=local_dataset_root)
                        except Exception as e:
                            sys.stdout.write(f"Exception raised, continuing in while loop. Exception: {e}")
                            time.sleep(10)
                            continue
                        sys.stdout.write("Local data storage: Copying done.\n")
                        # Create the doneFile
                        open(doneFile, mode='a').close()

                        # Remove the busyFile
                        os.remove(busyFile)

                    wait_it += 1
                    if wait_it == 360:
                        # Waited 1 hour, this should be done by now, check for errors
                        raise Exception("Waiting time too long. Check for errors.")

        elif is_coder:
            # We are running on a coder node and hence load the dataset directly from the permanent cache on z1
            dataset_root = os.path.join(permanent_cache, dataset_name)
        else:
            # Use the permanent dataset without copying
            dataset_root = permanent_dataset_root
        return dataset_root

    def get_dataloaders(self):
        base_dataset_name = self.config.dataset
        if 'same' in base_dataset_name:
            # Raise a warning that the validation and train datasets are not independent
            raise NotImplementedError("Validation and train datasets are not independent for same dataset, there might be overlap.")
        if self.requires_both_random_and_same:
            base_dataset_name = self.config.dataset.replace('both', 'random')
        rootPath = self.get_dataset_root(dataset_name=base_dataset_name)
        print(f"Loading {base_dataset_name} dataset from {rootPath}.")

        use_weighted_sampler = self.config.use_weighted_sampler or False
        assert not use_weighted_sampler, "Weighted sampler not implemented yet."

        scale_adjustments = {
            'scale_adjust_1234': self.config.scale_adjust_1234,
            'scale_adjust_6789': self.config.scale_adjust_6789,
            'scale_adjust_0': self.config.scale_adjust_0,
            'scale_adjust_51011': self.config.scale_adjust_51011,
        }

        trainData = SatelliteImageDataset(data_path=rootPath, shift_year=self.shift_year, 
                                          collapse_months=self.config.collapse_months,
                                          single_month_scaling=self.config.single_month_scaling,
                                          scale_adjustments=scale_adjustments, 
                                          is_3d_model=self.is_3d_model,
                                          time_mode=self.config.time_mode,
                                          has_sentinel_1=self.has_sentinel_1)

        # Select at maximum 10% of the data for validation using a fixed random seed. The total amount should be capped as 300*len(self.hardcoded_base_years)
        cut_off = 300*len(self.hardcoded_base_years)
        if self.debug:
            cut_off = int(0.1 * cut_off)
        n_val = min(int(0.1 * len(trainData)), cut_off)
        n_train = len(trainData) - n_val
        trainData, valData = torch.utils.data.random_split(trainData, [n_train, n_val], generator=torch.Generator().manual_seed(1234))

        if self.requires_both_random_and_same:
            # We now also need to load/copy the 'same' dataset and join it with trainData
            same_dataset_name = self.config.dataset.replace('both', 'same')
            same_rootPath = self.get_dataset_root(dataset_name=same_dataset_name)
            same_trainData = SatelliteImageDataset(data_path=same_rootPath, shift_year=self.shift_year, 
                                          collapse_months=self.config.collapse_months,
                                          single_month_scaling=self.config.single_month_scaling,
                                          scale_adjustments=scale_adjustments, 
                                          is_3d_model=self.is_3d_model,
                                          time_mode=self.config.time_mode,
                                          has_sentinel_1=self.has_sentinel_1)
            
            # We subset this again to have the indices attribute, this is just a hack such that both datasets have the same class structure
            same_trainData = torch.utils.data.Subset(same_trainData, list(range(len(same_trainData))))
            
            sys.stdout.write(f"Joining random ({len(trainData)}) and same ({len(same_trainData)}) datasets.\n")

        # Now get the train dataset for the specific years
        if self.config.years in [None, 'None', 'none'] or (isinstance(self.config.years, list) and (None in self.config.years or 'None' in self.config.years or 'none' in self.config.years)):
            years = self.hardcoded_base_years
        elif isinstance(self.config.years, int):
            years = [self.config.years]
        else:
            years = self.config.years
        sys.stdout.write(f"Filtering training data to contain only the following years: {years}.\n")
        datasets = [trainData, same_trainData] if self.requires_both_random_and_same else [trainData]
        
        for dataset in datasets:
            # trainData is now of type Subset, hence we have the attributes 'dataset' and 'indices'
            # We need to restrict the indices to the specific years by dropping all indices in 'indices' where the corresponding dataset.year_data is not in years
            dataset.indices = [i for i in dataset.indices if dataset.dataset.year_data[i] in years]
            # Note: we do not do this for valData, as we want to have validation metrics for all years independently

        if len(datasets) > 1:
            # Join trainData and same_trainData
            trainData = torch.utils.data.ConcatDataset([trainData, same_trainData])

        sys.stdout.write(f"Length of train and val splits: {len(trainData)}, {len(valData)}.\n")

        num_workers_default = self.config.num_workers_per_gpu if self.config.num_workers_per_gpu is not None else 8
        num_workers = num_workers_default * torch.cuda.device_count() * int(not self.debug)
        prefetch_factor = self.config.prefetch_factor   # This includes the None case, which defaults to 2 and is also needed when using num_workers = 0
        sys.stdout.write(f"Using {num_workers} workers.\n")
        train_sampler = None
        shuffle = True
        if use_weighted_sampler:
            train_sampler = WeightedRandomSampler(trainData.weights, len(trainData))
            shuffle = False

        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=self.config.batch_size, sampler=train_sampler,
                                                  pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                                                  shuffle=shuffle, prefetch_factor=prefetch_factor)
        valLoader = torch.utils.data.DataLoader(valData, batch_size=self.config.batch_size, shuffle=False,
                                                pin_memory=torch.cuda.is_available(), num_workers=2, prefetch_factor=prefetch_factor)


        return trainLoader, valLoader

    def get_model(self, reinit: bool, model_path: Optional[str] = None) -> torch.nn.Module:
        """
        Returns the model.
        :param reinit: If True, the model is reinitialized.
        :type reinit: bool
        :param model_path: Path to the model.
        :type model_path: Optional[str]
        :return: The model.
        :rtype: torch.nn.Module
        """
        print(
            f"Loading model - reinit: {reinit} | path: {model_path if model_path else 'None specified'}.")
        if reinit:
            # Define the model
            arch = self.config.arch or 'unet'
            backbone = self.config.backbone or 'resnet50'

            # The dataset must end in the format _months_channels_width_height, e.g. _6_12_256_256
            months, _, _, _ = self.config.dataset.split('_')[-4:]
            months = int(months)
            sys.stdout.write(f"Using {months} months.\n")
            assert months in [6, 12], f"Only 6 and 12 months are supported, not {months}."
            n_bands = 16 if self.has_sentinel_1 else 12
            if self.is_3d_model:
                in_channels = (months, n_bands + 1 if self.config.time_mode == 'channel' else n_bands)
            else:
                in_channels = n_bands if self.config.collapse_months else months*n_bands
                # Adjust in_channels if time_mode is 'channel'
                if self.config.time_mode == 'channel':
                    in_channels += 1  # Add an additional channel for the time variable
            sys.stdout.write(f"Using architecture {arch} with {in_channels} input channels, collapse_months = {self.config.collapse_months}, time_mode = {self.config.time_mode}.\n")
            assert arch in ['unet', 'single_year_unet3d','multi_year_unet3d']

            assert not (self.is_3d_model and self.config.collapse_months), "3D models are not compatible with collapse_months = True."
            if arch == 'unet':
                network_config = {
                    "encoder_name": backbone,
                    "encoder_weights": None if not self.config.use_pretrained_model else 'imagenet',
                    "in_channels": in_channels,
                    "classes": 1,
                }
                model = smp.Unet(**network_config)
            elif arch == 'single_year_unet3d':
                months, _, _, _ = self.config.dataset.split('_')[-4:]
                months = int(months)
                if months == 6:
                    model = SingleYearUNetSixMonth(n_channels=in_channels[1])
                elif months == 12:
                    model = SingleYearUNetTwelveMonth(n_channels=in_channels[1])
            if self.config.time_mode == 'rescale':
                model = RescaleHeadWrapper(model)
        else:
            # The model has been initialized already
            model = self.model

        if model_path is not None:
            # Load the model
            state_dict = torch.load(model_path, map_location=self.device)

            new_state_dict = OrderedDict()
            require_DP_format = isinstance(model,
                                           torch.nn.DataParallel)  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k
                new_state_dict[name] = v
            # Load the state_dict
            model.load_state_dict(new_state_dict)

        if self.dataParallel and reinit and not isinstance(model, torch.nn.DataParallel):
            # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)
        return model

    def get_loss(self, loss_name: str, threshold: float = None):
        assert loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber'], f"Loss {loss_name} not implemented."
        if threshold is not None:
            assert loss_name == 'l1', f"Threshold only implemented for l1 loss, not {loss_name}."
        # Dim 1 is the channel dimension, 0 is batch.
        # Sums up to get average height, could be mean without zeros
        remove_sub_track = lambda out, target: (out, torch.sum(target, dim=1))

        if loss_name == 'shift_l1':
            from losses.shift_l1_loss import ShiftL1Loss
            loss = ShiftL1Loss(ignore_value=0)
        elif loss_name == 'shift_l2':
            from losses.shift_l2_loss import ShiftL2Loss
            loss = ShiftL2Loss(ignore_value=0)
        elif loss_name == 'shift_huber':
            from losses.shift_huber_loss import ShiftHuberLoss
            loss = ShiftHuberLoss(ignore_value=0)
        elif loss_name == 'l1':
            from losses.l1_loss import L1Loss
            # Rescale the threshold to account for the label rescaling
            if threshold is not None:
                threshold = threshold
            loss = L1Loss(ignore_value=0, pre_calculation_function=remove_sub_track, lower_threshold=threshold)
        elif loss_name == 'l2':
            from losses.l2_loss import L2Loss
            loss = L2Loss(ignore_value=0, pre_calculation_function=remove_sub_track)
        elif loss_name == 'huber':
            from losses.huber_loss import HuberLoss
            loss = HuberLoss(ignore_value=0, pre_calculation_function=remove_sub_track, delta=3.0)
        loss = loss.to(device=self.device)
        return loss

    def get_visualization(self, viz_name: str, inputs, labels, outputs):
        assert viz_name in ['input_output', 'density_scatter_plot',
                            'boxplot'], f"Visualization {viz_name} not implemented."

        # Detach and copy the labels and outputs, then undo the rescaling
        labels, outputs = labels.detach().clone(), outputs.detach().clone()


        def remove_sub_track_vis(inputs, labels, outputs):
            return inputs, labels.sum(
                axis=1), outputs  # Same as remove_sub_track, but for visualization (i.e. has outputs as well)

        if viz_name == 'input_output':
            viz_fn = visualization.get_input_output_visualization(rgb_channels=[3,2,1],
                                                                      process_variables=remove_sub_track_vis, single_month_scaling=self.config.single_month_scaling)
        elif viz_name == 'density_scatter_plot':
            viz_fn = visualization.get_density_scatter_plot_visualization(ignore_value=0,
                                                                          process_variables=remove_sub_track_vis)
        elif viz_name == 'boxplot':
            viz_fn = visualization.get_visualization_boxplots(ignore_value=0, process_variables=remove_sub_track_vis)
        return viz_fn(inputs=inputs, labels=labels, outputs=outputs)

    def get_optimizer(self, initial_lr: float) -> torch.optim.Optimizer:
        """
        Returns the optimizer.
        :param initial_lr: The initial learning rate
        :type initial_lr: float
        :return: The optimizer.
        :rtype: torch.optim.Optimizer
        """
        wd = self.config['weight_decay'] or 0.
        optim_name = self.config.optim or 'AdamW'
        if optim_name == 'SGD':
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=initial_lr,
                                        momentum=0.9,
                                        weight_decay=wd,
                                        nesterov=wd > 0.)
        elif optim_name == 'AdamW':
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=initial_lr,
                                          weight_decay=wd)
        else:
            raise NotImplementedError

        return optimizer

    def save_model(self, model_identifier: str, sync: bool = False) -> str:
        """
        Saves the model's state_dict to a file.
        :param model_identifier: Name of the file type.
        :type model_identifier: str
        :param sync: Whether to sync the file to wandb.
        :type sync: bool
        :return: Path to the saved model.
        :rtype: str
        """
        fName = f"{model_identifier}_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)

        # Only save models in their non-module version, to avoid problems when loading
        try:
            model_state_dict = self.model.module.state_dict()
        except AttributeError:
            model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict

        if sync:
            wandb.save(fPath)
        return fPath

    def log(self, step: int, phase_runtime: float, commit: bool = True):
        """
        Logs the current training status.
        :param phase_runtime: The wall-clock time of the current phase.
        :type phase_runtime: float
        """
        loggingDict = self.get_metrics()
        loggingDict.update({
            'phase_runtime': phase_runtime,
            'iteration': step,
            'samples_seen': step * self.config.batch_size,
        })

        # Log and push to Wandb
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"{metric_type}"] = val

        wandb.log(loggingDict, commit=commit)
    
    def log_initial_state(self):
        # Log the initial state of the model before any training iteration
        phase_runtime = 0  # No runtime yet
        step = 0  # Initial step
        x_input, y_target, year = next(iter(self.loader['train']))
        x_input = x_input.to(device=self.device, non_blocking=True)
        y_target = y_target.to(device=self.device, non_blocking=True)
        year = year.to(device=self.device, non_blocking=True)
        if self.config.time_mode == 'rescale':
            output = self.model.eval()(x_input, year)
        else:
            output = self.model.eval()(x_input)

        # Create the visualizations
        for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
            viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
            wandb.log({'train/' + viz_func: wandb.Image(viz)}, commit=False)

        if not self.debug:
            # Evaluate the validation dataset
            self.eval(data='val')

        self.log(step=step, phase_runtime=phase_runtime)
        self.reset_averaged_metrics()

    def define_optimizer_scheduler(self):
        # Define the optimizer
        initial_lr = self.config.initial_lr
        self.optimizer = self.get_optimizer(initial_lr=initial_lr)

        # We define a scheduler. All schedulers work on a per-iteration basis
        n_total_iterations = self.config.n_iterations
        n_warmup_iterations = int(0.1 * n_total_iterations)

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if n_warmup_iterations > 0:
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                      start_factor=1.0, end_factor=0.,
                                                      total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    @torch.no_grad()
    def eval(self, data: str):
        """
        Evaluates the model on the given data set.
        :param data: string indicating the data set to evaluate on. Can be 'train', 'val', or 'best_val'.
        :type data: str
        """
        if data == 'best_val':
            sys.stdout.write(f"Evaluating early stopping model on validation split.\n")
            dataloader_id = 'val'
        else:
            sys.stdout.write(f"Evaluating on {data} split.\n")
            dataloader_id = data
        for step, batch in enumerate(tqdm(self.loader[dataloader_id]), 1):
            x_input, y_target, year = batch
            x_input = x_input.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)
            year = year.to(device=self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                if self.config.time_mode == 'rescale':
                    output = self.model.eval()(x_input, year)
                else:
                    output = self.model.eval()(x_input)
                loss = self.loss_criteria[self.loss_name](output, y_target)
                self.metrics[data]['loss'](value=loss, weight=len(y_target))
                for loss_type in self.loss_criteria.keys():
                    metric_loss = self.loss_criteria[loss_type](output, y_target)
                    # Check if the metric_loss is nan
                    if not torch.isnan(metric_loss):
                        self.metrics[data][loss_type](value=metric_loss, weight=len(y_target))
                
                # Compute the loss for each year
                for current_year in self.hardcoded_base_years:
                    for loss_name in self.metrics[f'val_{current_year}'].keys():
                        if loss_name == 'loss':
                            loss_fn = self.loss_criteria[self.loss_name]    # self.loss_name is the name of the loss function
                        else:
                            loss_fn = self.loss_criteria[loss_name]
                        # Create a mask for the current year
                        mask = (year == (current_year-self.shift_year)).flatten()
                        metric_loss = loss_fn(output[mask], y_target[mask])
                        # Check if the metric_loss is nan
                        if not torch.isnan(metric_loss):
                            self.metrics[f'val_{current_year}'][loss_name](value=metric_loss, weight=len(y_target[mask]))

            if step <= 4:
                # Create the visualizations for the first 4 batches
                for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
                    wandb.log({data + '/' + viz_func + "_" + str(step): wandb.Image(viz)}, commit=False)
            torch.cuda.empty_cache()    # Might help with memory issues according to this thread: https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354


    def train(self):
        log_freq, n_iterations = self.config.log_freq, self.config.n_iterations
        ampGradScaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.reset_averaged_metrics()
        phase_start = time.time()

        # Determine log steps
        if self.debug:
            log_steps = []
        elif isinstance(log_freq, float) and 0 < log_freq < 1:
            log_steps = set(int(i * n_iterations * log_freq) for i in range(1, int(1 / log_freq) + 1))
        else:
            log_steps = set(range(log_freq, n_iterations + 1, log_freq))

        # Initial logging before any training iteration
        if not self.debug:
            self.log_initial_state()

        for step in tqdm(range(1, n_iterations + 1, 1)):
            # Reinitialize the train iterator if it reaches the end
            if step == 1 or (step - 1) % len(self.loader['train']) == 0:
                train_iterator = iter(self.loader['train'])

            # Move to CUDA if possible
            batch = next(train_iterator)
            x_input, y_target, year = batch
            x_input = x_input.to(device=self.device, non_blocking=True)
            y_target = y_target.to(device=self.device, non_blocking=True)

            if self.config.time_mode == 'rescale':
                year = year.to(device=self.device, non_blocking=True)
            else:
                year = None

            self.optimizer.zero_grad()

            itStartTime = time.time()
            with autocast(enabled=self.use_amp):
                if year is not None:
                    output = self.model.train()(x_input, year)
                else:
                    output = self.model.train()(x_input)
                loss = self.loss_criteria[self.loss_name](output, y_target)
                ampGradScaler.scale(loss).backward()  # Scaling + Backpropagation
                # Unscale the weights manually, normally this would be done by ampGradScaler.step(), but since
                # we might use gradient clipping, this has to be split
                ampGradScaler.unscale_(self.optimizer)
                if self.config.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                ampGradScaler.step(self.optimizer)
                ampGradScaler.update()  # This should happen only once per iteration
                self.scheduler.step()
                self.metrics['train']['loss'](value=loss, weight=len(y_target))

                with torch.no_grad():
                    for loss_type in self.loss_criteria.keys():
                        metric_loss = self.loss_criteria[loss_type](output, y_target)
                        # Check if the metric_loss is nan
                        if not torch.isnan(metric_loss):
                            self.metrics['train'][loss_type](value=metric_loss, weight=len(y_target))
                itEndTime = time.time()
                n_img_in_iteration = int(self.config.batch_size)
                ips = n_img_in_iteration / (itEndTime - itStartTime)  # Images processed per second
                self.metrics['train']['ips_throughput'](ips)

            if step in log_steps or step == n_iterations:
                phase_runtime = time.time() - phase_start
                # Create the visualizations
                for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
                    wandb.log({'train/' + viz_func: wandb.Image(viz)}, commit=False)

                # Evaluate the validation dataset
                if not self.debug:
                    self.eval(data='val')
                current_val_loss = self.metrics['val']['loss'].compute()

                commit = step < n_iterations or (not self.use_early_stopping)
                self.log(step=step, phase_runtime=phase_runtime, commit=commit)
                self.reset_averaged_metrics()
                phase_start = time.time()

                # Check for early stopping
                if self.use_early_stopping:
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.best_model_path = self.save_model(model_identifier='best', sync=False)

        # Reload the best model if early stopping was used
        if self.use_early_stopping and self.best_model_path is not None:
            self.model = self.get_model(reinit=False, model_path=self.best_model_path)
            self.eval(data='best_val')
            self.log_best_model_metrics()

    def log_best_model_metrics(self):
        """Logs the best model's metrics after training."""
        loggingDict = {}
        # Add metrics
        split = 'best_val'
        for metric_name, metric in self.metrics[split].items():
            loggingDict[f"{split}/{metric_name}"] = metric.compute()
            # Add the metrics for each year
            for year in self.hardcoded_base_years:
                for metric_name, metric in self.metrics[f'best_val_{year}'].items():
                    loggingDict[f'best_val_{year}/{metric_name}'] = metric.compute()

        # Log and push to Wandb
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"{metric_type}"] = val

        wandb.log(loggingDict, commit=True) # Now we commit

    def run(self):
        """Controls the execution of the script."""
        # We start training from scratch
        self.set_seed(seed=self.seed)  # Set the seed
        loaders = self.get_dataloaders()
        self.loader['train'], self.loader['val'] = loaders
        self.model = self.get_model(reinit=True, model_path=self.model_paths['initial'])  # Load the model

        self.define_optimizer_scheduler()  # This was moved before define_strategy to have the optimizer available

        self.train()  # Train the model

        # Save the trained model and upload it to wandb
        self.save_model(model_identifier='trained', sync=True)

