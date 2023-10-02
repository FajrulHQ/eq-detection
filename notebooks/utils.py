import torch
import os, json
import numpy as np
import scipy
from pathlib import Path as _Path

import seisbench
from seisbench.util import worker_seeding
from seisbench import config
from seisbench.generate import PickLabeller, SupervisedLabeller
from seisbench.generate.labeling import gaussian_pick, triangle_pick, box_pick

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchsummary import summary

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib

class Solver:
    def __init__(self, train_generator, test_generator, batch_size=64, num_workers=2, folder_name='test'):
        self.train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)
        self.test_loader = DataLoader(test_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)
        self.history = dict(
            train_loss=[], train_f1_d=[], train_f1_p=[], train_f1_s=[],
            test_loss=[], test_f1_d=[], test_f1_p=[], test_f1_s=[]
        )
        self.folder_path = _Path(f'./outputs/{folder_name}')
        self.checkpoint_path = self.folder_path / 'checkpoint'
        self.input_size = (batch_size, 3, 6000)
        # self.writer = SummaryWriter()
        
        os.makedirs(self.folder_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def train(self, 
              model,  
              learning_rate=1e-2, 
              epochs=50, 
              class_weights=[0.05, 0.40, 0.55],
              print_every_batch=15,
              patience=12
             ):
        self.model = model
        self.class_weights = class_weights
        self.print_every_batch = print_every_batch
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.patience = patience
        self.current_patience = 0
        self.best_metric = np.Inf
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=.5, verbose=True)

        for t in range(epochs):
            print(f"\nEpoch {t+1}, Learning Rate: {self.optimizer.param_groups[0]['lr']}\n-------------------------------")
            train_loss, train_f1 = self.train_loop(self.train_loader)
            test_loss, test_f1 = self.test_loop(self.test_loader)
            # self.writer.add_scalar('Train Loss', train_loss, global_step=t)
            # self.writer.add_scalar('Validation Loss', test_loss, global_step=t)
            # self.writer.add_scalar('F1 Score', test_f1, global_step=t)
            
            # Log histograms
            # for name, param in self.model.named_parameters():
            #     self.writer.add_histogram(name, param, global_step=t)
            
            torch.save(self.model.state_dict(), self.checkpoint_path / f'model_checkpoint_{t+1}.h5')

            self.save_history()
            # early stopping
            if self.current_patience >= self.patience:
                print(f'Early stopping after {t+1} epochs')
                break
        
        torch.save(self.model.state_dict(), self.folder_path / f'model_final.h5')
        self.save_summary()
        # self.writer.close()
            
    def summary(self):
        return summary(self.model, (3,6000))

    def save_history(self):
        matplotlib.use('Agg')
        fig, ax = plt.subplots(1,4, figsize=(16,4))
        for key, val in self.history.items():
            x = [i+1 for i in range(len(val))]
            if 'loss' in key:   ax[0].plot(x, val, '.--', label=key); ax[0].set_title('loss function')
            elif '_d' in key:   ax[1].plot(x, val, '.--', label=key); ax[1].set_title('detection')
            elif '_p' in key:   ax[2].plot(x, val, '.--', label=key); ax[2].set_title('picker_p')
            else:               ax[3].plot(x, val, '.--', label=key); ax[3].set_title('picker_s')
            
        for i in range(4):
            ax[i].set_ylabel('F1 Score')
            ax[i].set_xlabel('epoch')
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            if i>0: ax[i].set_ylim(0,1)
            ax[i].legend()
        ax[0].set_ylabel('Loss')
        plt.tight_layout()
        
        plt.savefig(self.folder_path / 'history.png')
        with open(self.folder_path / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def save_summary(self):
        f1_d = self.history['test_f1_d'][-1]*self.class_weights[0]
        f1_p = self.history['test_f1_p'][-1]*self.class_weights[1]
        f1_s = self.history['test_f1_s'][-1]*self.class_weights[2]
        f1 = (f1_d+f1_p+f1_s)

        summary_text = f"Model Summary:\n"
        summary_text += f"-----------------------------\n"
        summary_text += f"F1 Score\n"
        for label, title in zip(['f1_d', 'f1_p', 'f1_s'], ['detector', 'picker_p', 'picker_s']):
            score = self.history[f'test_{label}'][-1]
            summary_text += f"\t{title}\t\t\t: {score:.4f}\n"
        summary_text += f"Total F1 Score\t: {f1:.4f}\n"

        with open(self.folder_path / 'summary.txt', 'w') as f:
            f.write(summary_text)

    def _single_loss_fn(self, y_pred, y_true, eps=1e-5):
        y_true = (y_true > 0.5).float().clone()
        h = y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps)
        # h = y_true * torch.log(y_pred + eps)
        # h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h

    def _single_f1_score(self, y_pred_, y_true_, is_phase=False):
        # Threshold predictions to get binary labels
        thres = 0.3 if is_phase else 0.5
        y_pred = (y_pred_ > thres).float().clone()
        y_true = (y_true_ > thres).float().clone()

        # Calculate true positives, false positives, and false negatives for each sample in the batch
        tp = (y_pred * y_true).sum(dim=1)
        fp = (y_pred * (1 - y_true)).sum(dim=1)
        fn = ((1 - y_pred) * y_true).sum(dim=1)

        # Calculate precision and recall for each sample in the batch
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        # Calculate F1 score for each sample in the batch
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        # Take the average F1 score across all samples in the batch
        f1 = f1.mean()

        return f1

    def loss_fn(self, y_pred, y_true, eps=1e-5):
        '''
        Params
        ------

        y_pred: tuple of EQTransformer output, i.e. (pred_d, pred_p, pred_s)
        y_true: torch Tensor of shape (batch_size, 3, seq_len)
            y_true[:, k, :] for k = 0, 1, 2 is for true_d, true_p, and true_s respectively

        Returns
        -------
            (loss, loss_d, loss_p, loss_s)
        '''
        loss_d = self._single_loss_fn(y_pred[0], y_true[:, 0, :])
        loss_p = self._single_loss_fn(y_pred[1], y_true[:, 1, :])
        loss_s = self._single_loss_fn(y_pred[2], y_true[:, 2, :])
        
        loss = 0
        for weight, loss_ in zip(self.class_weights, [loss_d, loss_p, loss_s]):
            loss += (weight * loss_)

        return (loss, loss_d, loss_p, loss_s)

    def f1_score(self, y_true, y_pred):
        f1_d = self._single_f1_score(y_pred[0], y_true[:, 0, :])
        f1_p = self._single_f1_score(y_pred[1], y_true[:, 1, :], is_phase=True)
        f1_s = self._single_f1_score(y_pred[2], y_true[:, 2, :], is_phase=True)
        
        return (f1_d, f1_p, f1_s)

    def train_loop(self, dataloader):
        num_batches = len(dataloader)
        train_loss, train_f1_d, train_f1_p, train_f1_s = 0,0,0,0
        for batch_id, batch in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.model(batch["X"].to(self.model.device))
            true = batch["y"].to(self.model.device)
            loss, loss_d, loss_p, loss_s = self.loss_fn(pred, true)
            f1_d, f1_p, f1_s = self.f1_score(true, pred)

            train_loss += loss
            train_f1_d += f1_d
            train_f1_p += f1_p
            train_f1_s += f1_s

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_id % self.print_every_batch == 0:
                loss, current = loss.item(), (batch_id+1) * batch["X"].shape[0]
                met = f"[{current:>5d}/{len(dataloader.dataset):>5d}] "
                met += ' | '.join([
                    f"loss: {loss:>4f}",
                    f"loss_d: {loss_d:>4f}",
                    f"loss_p: {loss_p:>4f}",
                    f"loss_s: {loss_s:>4f}",
                    f"f1_d: {f1_d:>2f}",
                    f"f1_p: {f1_p:>2f}",
                    f"f1_s: {f1_s:>2f}",
                ])
                print(met)

        self.history['train_loss'].append(train_loss.item() / num_batches)
        self.history['train_f1_d'].append(train_f1_d.item() / num_batches)
        self.history['train_f1_p'].append(train_f1_p.item() / num_batches)
        self.history['train_f1_s'].append(train_f1_s.item() / num_batches)

        train_f1 = sum([f1*cw for f1,cw in zip([train_f1_d,train_f1_p,train_f1_s], self.class_weights)])
        return train_loss, train_f1

    def test_loop(self, dataloader):
        num_batches = len(dataloader)
        test_loss, test_loss_d, test_loss_p, test_loss_s = 0, 0, 0, 0
        test_f1_d, test_f1_p, test_f1_s = 0, 0, 0

        with torch.no_grad():
            for batch in dataloader:
                pred = self.model(batch["X"].to(self.model.device))
                true = batch["y"].to(self.model.device)
                loss, loss_d, loss_p, loss_s = self.loss_fn(pred, true)
                f1_d, f1_p, f1_s = self.f1_score(true, pred)
                
                test_loss_d += loss_d
                test_loss_p += loss_p
                test_loss_s += loss_s
                test_loss   += loss
                test_f1_d   += f1_d
                test_f1_p   += f1_p
                test_f1_s   += f1_s

        test_loss   = test_loss.item()   / num_batches
        test_loss_d = test_loss_d.item() / num_batches
        test_loss_p = test_loss_p.item() / num_batches
        test_loss_s = test_loss_s.item() / num_batches
        test_f1_d   = test_f1_d.item()   / num_batches
        test_f1_p   = test_f1_p.item()   / num_batches
        test_f1_s   = test_f1_s.item()   / num_batches

        if test_loss < self.best_metric:
            self.best_metric = test_loss
            self.current_patience = 0
        else: 
            self.current_patience+=1 
        
        self.history['test_loss'].append(test_loss)
        self.history['test_f1_d'].append(test_f1_d)
        self.history['test_f1_p'].append(test_f1_p)
        self.history['test_f1_s'].append(test_f1_s)

        # Adabtive learning rate scheduler
        self.scheduler.step(test_loss)
        
        met = f"[test] "
        met += ' | '.join([
            f"loss: {test_loss:>4f}",
            f"loss_d: {test_loss_d:>4f}",
            f"loss_p: {test_loss_p:>4f}",
            f"loss_s: {test_loss_s:>4f}",
            f"f1_d: {test_f1_d:>2f}",
            f"f1_p: {test_f1_p:>2f}",
            f"f1_s: {test_f1_s:>2f}",
        ])
        print(met)

        test_f1 = sum([f1*cw for f1,cw in zip([test_f1_d,test_f1_p,test_f1_s], self.class_weights)])
        return test_loss, test_f1

class DetectionLabeller(SupervisedLabeller):
    """
    Create detection labels from picks.
    The labeler can either use fixed detection length or determine the length from the P to S time as in
    Mousavi et al. (2020, Nature communications). In the latter case, detections range from P to S + factor * (S - P)
    and are only annotated if both P and S phases are present.
    All detections are represented through a boxcar time series with the same length as the input waveforms.
    For both P and S, lists of phases can be passed of which the sequentially first one will be used.
    All picks with NaN sample are treated as not present.

    :param p_phases: (List of) P phase metadata columns
    :type p_phases: str, list[str]
    :param s_phases: (List of) S phase metadata columns
    :type s_phases: str, list[str]
    :param factor: Factor for length of window after S onset
    :type factor: float
    :param fixed_window: Number of samples for fixed window detections. If none, will determine length from P to S time.
    :type fixed_window: int
    """

    def __init__(
        self, p_phases, s_phases=None, factor=1.4, fixed_window=None, spectrogram_based=False, **kwargs
    ):
        self.label_method = "probabilistic"
        self.label_columns = "detections"
        if isinstance(p_phases, str):
            self.p_phases = [p_phases]
        else:
            self.p_phases = p_phases

        if isinstance(s_phases, str):
            self.s_phases = [s_phases]
        elif s_phases is None:
            self.s_phases = []
        else:
            self.s_phases = s_phases

        if s_phases is not None and fixed_window is not None:
            seisbench.logger.warning(
                "Provided both S phases and fixed window length to DetectionLabeller. "
                "Will use fixed window size and ignore S phases."
            )

        self.factor = factor
        self.fixed_window = fixed_window
        self.spectrogram_based = spectrogram_based

        kwargs["dim"] = kwargs.get("dim", -2)
        super().__init__(label_type="multi_class", **kwargs)

    def get_spectrogram(self, data, winlen=16):
        f,t, Sp = scipy.signal.spectrogram(data, 40, nperseg=winlen)
        pSp = [np.mean(s) for s in Sp.T]

        f = scipy.interpolate.interp1d(t, pSp, kind='linear')
        x = np.linspace(t[0],t[-1],6000)
        y = f(x)
        return y
    
    def calculate_coda(self, data, p_arrival, s_arrival, p_next=6000):
        win_mavg = (p_next-p_arrival)//4
        sp = np.array([self.get_spectrogram(x) for x in data ])
        mask = np.zeros(sp.shape)
        mask[:, p_arrival:p_next] = 1
        sp = sp*mask

        en_smooth = np.zeros_like(data[:, p_arrival:p_next])
        for i in range(len(data)):
            sp_ = sp[i , p_arrival:p_next]
            cum_en = np.array([sp_[s:].sum() for s in range(len(sp_))])
            en_smooth[i] = np.convolve(cum_en, np.ones(win_mavg)/win_mavg, mode='same')

        std_mean = en_smooth.mean(axis=0)
        diff = np.diff(std_mean, 2)
        diff[diff < 0] = 0

        std_mean_norm = (std_mean-std_mean.min())/(std_mean.max()-std_mean.min())
        coda_ = np.argmax(diff) + 2 + p_arrival
        snr = 0
        
        while (coda_<p_next and snr<2):
            if coda_>s_arrival:
                sc = data[:, s_arrival:coda_].mean(axis=0)
                nc = data[:, coda_:coda_+(coda_-s_arrival)].mean(axis=0)
                snr = sc.std()/nc.std()
                # print(snr)
            coda_ += 40
            if coda_>p_next-1: coda_=p_next

        return coda_, std_mean_norm

    def label(self, X, metadata):
        sample_dim, channel_dim, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        if self.fixed_window:
            # Only label until end of fixed window
            factor = 0
        else:
            factor = self.factor
    
        if self.ndim == 2:
            y = np.zeros((1, X.shape[width_dim]))
            p_arrivals = [
                metadata[phase]
                for phase in self.p_phases
                if phase in metadata and not np.isnan(metadata[phase])
            ]
            if self.fixed_window is not None:
                # Fake S arrivals for simulating fixed window
                s_arrivals = [x + self.fixed_window for x in p_arrivals]
            else:
                s_arrivals = [
                    metadata[phase]
                    for phase in self.s_phases
                    if phase in metadata and not np.isnan(metadata[phase])
                ]

            if len(p_arrivals) != 0 and len(s_arrivals) != 0:
                p_arrival = min(p_arrivals)
                s_arrival = min(s_arrivals)
                p_to_s = s_arrival - p_arrival
                if s_arrival >= p_arrival:
                    if self.spectrogram_based:
                        p0 = max(int(p_arrival), 0)
                        s0 = max(int(s_arrival), 0)
                        p1, _ = self.calculate_coda(X, p0, s0)
                        y[0, p0:p1] = 1
                    # Only annotate valid options
                    else:
                        p0 = max(int(p_arrival), 0)
                        p1 = max(int(s_arrival + factor * p_to_s), 0)
                        y[0, p0:p1] = 1

        elif self.ndim == 3:
            y = np.zeros(
                shape=(
                    X.shape[sample_dim],
                    1,
                    X.shape[width_dim],
                )
            )
            p_arrivals = [
                metadata[phase] for phase in self.p_phases if phase in metadata
            ]

            if self.fixed_window is not None:
                # Fake S arrivals for simulating fixed window
                s_arrivals = [x + self.fixed_window for x in p_arrivals]
            else:
                s_arrivals = [
                    metadata[phase] for phase in self.s_phases if phase in metadata
                ]

            if len(p_arrivals) != 0 and len(s_arrivals) != 0:
                p_arrivals = np.stack(p_arrivals, axis=-1)  # Shape (samples, phases)
                s_arrivals = np.stack(s_arrivals, axis=-1)

                mask = np.logical_and(
                    np.any(~np.isnan(p_arrivals), axis=1),
                    np.any(~np.isnan(s_arrivals), axis=1),
                )
                if not mask.any():
                    return y

                p_arrivals = np.nanmin(
                    p_arrivals[mask, :], axis=1
                )  # Shape (samples (which are present),)
                s_arrivals = np.nanmin(s_arrivals[mask, :], axis=1)
                p_to_s = s_arrivals - p_arrivals

                starts = p_arrivals.astype(int)
                ends = (s_arrivals + factor * p_to_s).astype(int)

                # print(mask, starts, ends)
                for i, s, e in zip(np.arange(len(mask))[mask], starts, ends):
                    s = max(0, s)
                    e = max(0, e)
                    y[i, 0, s:e] = 1

        else:
            raise ValueError(
                f"Illegal number of input dimensions for DetectionLabeller (ndim={self.ndim})."
            )

        return y

    def __str__(self):
        return f"DetectionLabeller (label_type={self.label_type}, dim={self.dim})"


class ProbabilisticLabeller(PickLabeller):
    r"""
    Create supervised labels from picks. The picks in example are represented
    probabilistically with shapes of:

    *  gaussian:

        .. math::
           X \sim \mathcal{N}(\mu,\,\sigma^{2})

    *  triangle::

           #         / \
           #        /   \
           #       /     \
           #      /       \
           #     /         \
           # ___/           \___
           #    ----- | -----
           #      2*sigma (sigma = half width)

    *  box::

           #        ------------
           #        |          |
           #        |          |
           #        |          |
           #---------          --------
           #        ---- | ----
           #         2*sigma (sigma = half width)

    All picks with NaN sample are treated as not present.
    The noise class is automatically created as :math:`\max \left(0, 1 - \sum_{n=1}^{c} y_{j} \right)`.

    :param sigma: Variance of Gaussian (gaussian), half-width of triangle ('triangle')
                or box function ('box') label representation in samples, defaults to 10.
    :type sigma: int, optional
    """

    def __init__(self, shape="gaussian", sigma=10, use_detection=False, spectrogram_based=False, **kwargs):
        self.label_method = "probabilistic"
        self.sigma = sigma
        self.shape = shape
        self.use_detection = use_detection
        self.spectrogram_based = spectrogram_based
        self._labelshape_fn_mapper = {
            "gaussian": gaussian_pick,
            "triangle": triangle_pick,
            "box": box_pick,
        }
        kwargs["dim"] = kwargs.get("dim", 1)
        super().__init__(label_type="multi_class", **kwargs)

    def label(self, X, metadata):
        if not self.label_columns:
            label_columns = self._auto_identify_picklabels(metadata)
            (
                self.label_columns,
                self.labels,
                self.label_ids,
            ) = self._colums_to_dict_and_labels(label_columns)

        sample_dim, channel_dim, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        if self.ndim == 2:
            y = np.zeros(shape=(len(self.labels), X.shape[width_dim]))
        elif self.ndim == 3:
            y = np.zeros(
                shape=(
                    X.shape[sample_dim],
                    len(self.labels),
                    X.shape[width_dim],
                )
            )

        # Construct pick labels
        for label_column, label in self.label_columns.items():
            i = self.label_ids[label]

            if label_column not in metadata:
                # Unknown pick
                continue

            if isinstance(metadata[label_column], (int, np.integer, float)):
                # Handle single window case
                onset = metadata[label_column]
                if self.shape in self._labelshape_fn_mapper.keys():
                    label_val = self._labelshape_fn_mapper[self.shape](
                        onset=onset, length=X.shape[width_dim], sigma=self.sigma
                    )
                else:
                    raise ValueError(
                        f"Labeller of shape {self.shape} is not implemented."
                    )

                label_val[
                    np.isnan(label_val)
                ] = 0  # Set non-present pick probabilities to 0
                y[i, :] = np.maximum(y[i, :], label_val)
            else:
                # Handle multi-window case
                for j in range(X.shape[sample_dim]):
                    onset = metadata[label_column][j]
                    if self.shape in self._labelshape_fn_mapper.keys():
                        label_val = self._labelshape_fn_mapper[self.shape](
                            onset=onset, length=X.shape[width_dim], sigma=self.sigma
                        )
                    else:
                        raise ValueError(
                            f"Labeller of shape {self.shape} is not implemented."
                        )

                    label_val[
                        np.isnan(label_val)
                    ] = 0  # Set non-present pick probabilities to 0
                    y[j, i, :] = np.maximum(y[j, i, :], label_val)

        y /= np.maximum(
            1, np.nansum(y, axis=channel_dim, keepdims=True)
        )  # Ensure total probability mass is at most 1

        # Construct noise label
        if self.ndim == 2:
            y[self.label_ids["Noise"], :] = 1 - np.nansum(y, axis=channel_dim)
            y = self._swap_dimension_order(
                y,
                current_dim="CW",
                expected_dim=config["dimension_order"].replace("N", ""),
            )
        elif self.ndim == 3:
            y[:, self.label_ids["Noise"], :] = 1 - np.nansum(y, axis=channel_dim)
            y = self._swap_dimension_order(
                y, current_dim="NCW", expected_dim=config["dimension_order"]
            )

        if (self.use_detection):
            if self.ndim == 2:
                p_phases = {k:v for k,v in self.label_columns.items() if v=='P'}
                s_phases = {k:v for k,v in self.label_columns.items() if v=='S'}
                dl = DetectionLabeller(p_phases=p_phases, s_phases=s_phases, spectrogram_based=self.spectrogram_based)
                dl.ndim = self.ndim
                self.label_type = 'multilabel'
            
                y[2, :] = dl.label(X, metadata)[0]
                y = y[[2,0,1], :]

        return y

    def __str__(self):
        return f"ProbabilisticLabeller (label_type={self.label_type}, dim={self.dim})"

