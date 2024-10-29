'''
Functions for FROC calculation
'''

# %%
import sklearn.metrics
import numpy as np
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Tuple, Union, Any


# %%
def preprocess_result_manifest(
    manifest: pd.DataFrame,
    filter: list = None
) -> pd.DataFrame:
    manifest = manifest.copy()
    if filter is not None:
        manifest = manifest[manifest['DatasetTag'].isin(filter)]

    # convert the bbox back to numpy
    bboxes = []
    detection_prob = []
    for v in manifest['bbox'].values:
        bbox = [float(s) for s in v[1:-1].split(' ') if len(s) > 0]
        bboxes.append(np.array(bbox))
        detection_prob.append(bbox[0])
    manifest['bbox'] = bboxes

    manifest['label'] = 1
    manifest.loc[manifest['Tag'].isin(['prediction']), 'label'] = 0

    return manifest


# %%
def get_froc_manifest(manifest, pred_cols=['prob'], iou_threshold=None):
    '''
    Each label aneurysm will take the largest prediction.
    Fns are set to 0.
    Fps are preserved.
    '''

    records = []
    for mrn in pd.unique(manifest.MRN):
        sub_df = manifest[manifest.MRN == mrn]

        # find if each annotations is predicted or not
        df_annotation = sub_df[sub_df.Tag == 'annotation']
        df_pred = sub_df[sub_df.Tag == 'prediction']
        for ianno, row_anno in df_annotation.iterrows():
            lbb = row_anno['bbox']
            for col in pred_cols:
                max_pred = -1
                for ipred, row_pred in df_pred.iterrows():
                    pbb = row_pred['bbox']
                    if iou_threshold is None:
                        # test if predict bbox's center is inside the lesion
                        if np.linalg.norm(lbb[1:4] - pbb[1:4]) <= lbb[4] / 2:
                            max_pred = max(max_pred, row_pred[col])
                            df_pred.at[ipred, 'label'] = 1
                    else:
                        raise NotImplementedError('IoU-based metric not implemented')
                df_annotation.at[ianno, col] = max_pred

        # preserve the annotations and false positives
        records.append(df_annotation)
        records.append(df_pred[df_pred['label'] == 0])
    records = pd.concat(records)

    return records.reset_index(drop=True)


# %%
def froc_with_ci(
    preds: np.array,
    labels: np.array,
    mrns: np.array,
    mrns_empty: np.array,
    ci: float = 95,
    nbst: int = 1000,
    seed: Union[int, np.random.Generator] = None,
    average_at_fp: List[float] = [0.125, 0.25, 0.5, 1, 2, 4, 8]
) -> dict:
    '''
    FROC curve with confidence interval

    @params
    @mrns: the mrn of each element in preds and labels
    @mrns_empty: the mrns of the patients that do not have label or prediction
    '''

    # make sure there is no overlap between mrns and mrns_empty
    assert(len(np.unique(mrns)) + len(np.unique(mrns_empty)) == len(np.unique(list(mrns) + list(mrns_empty))))

    fprs, tprs, ths = sklearn.metrics.roc_curve(labels, preds)
    npatients = len(np.unique(mrns)) + len(np.unique(mrns_empty))
    nfps = np.sum(1 - labels)
    fprs_per_patient = fprs * nfps / npatients
    # remove last element in the FROC
    fprs_per_patient = fprs_per_patient[:-1]
    tprs = tprs[:-1]
    ths = ths[:-1]
    # get the averaged tprs at fprs
    average_tprs = np.interp(average_at_fp, fprs_per_patient, tprs)
    average_tprs = average_tprs.mean()

    # bootstrap to get confidence interval
    tpr_bst_list = []
    average_tpr_bst_list = []
    # bootstrap by patients. So should get the preds and labels for each patient
    preds_patient = {}
    labels_patient = {}
    for mrn in np.unique(mrns):
        inds = np.where(mrns == mrn)
        preds_patient[mrn] = preds[inds]
        labels_patient[mrn] = labels[inds]
    # add the empty patients
    for mrn in np.unique(mrns_empty):
        preds_patient[mrn] = np.array([])
        labels_patient[mrn] = np.array([])
    unique_mrns = [m for m in preds_patient]

    rng = np.random.default_rng(seed)
    for i in range(nbst):
        # sampling with replacement
        sampled_mrns = rng.choice(unique_mrns, len(unique_mrns), True)
        label_bst = np.concatenate([labels_patient[m] for m in sampled_mrns])
        pred_bst = np.concatenate([preds_patient[m] for m in sampled_mrns])

        # calculate ROC for each bootstrap
        fpr_bst, tpr_bst, _ = sklearn.metrics.roc_curve(label_bst, pred_bst)
        nfps_bst = np.sum(1 - label_bst)
        fprs_per_patient_bst = fpr_bst * nfps_bst / npatients
        # remove last element in the FROC
        fprs_per_patient_bst = fprs_per_patient_bst[:-1]
        tpr_bst = tpr_bst[:-1]
        # get the averaged tprs at fprs
        average_tprs_bst = np.interp(average_at_fp, fprs_per_patient_bst, tpr_bst)
        average_tprs_bst = average_tprs_bst.mean()
        average_tpr_bst_list.append(average_tprs_bst)

        # resample the roc for each fprs
        tpr_sample = np.interp(fprs_per_patient, fprs_per_patient_bst, tpr_bst)
        tpr_bst_list.append(tpr_sample)

    tpr_bst_list = np.array(tpr_bst_list)
    average_tpr_bst_list = np.array(average_tpr_bst_list)
    tprs_ci = np.percentile(tpr_bst_list, [(100 - ci) / 2, 100 - (100 - ci) / 2], axis=0)
    average_tpr_ci = np.percentile(average_tpr_bst_list, [(100 - ci) / 2, 100 - (100 - ci) / 2])

    return {
        'fpr_per_patient': fprs_per_patient,
        'tprs': tprs,
        'ths': ths,
        'tprs_ci': tprs_ci,
        'average_tpr': average_tprs,
        'average_tpr_ci': average_tpr_ci
    }


def plot_roc_with_ci(
    ax: matplotlib.axes.Axes,
    fprs: np.array,
    tprs: np.array,
    tprs_ci: np.array,
    color: Any = None,
    linestyle: Any = None
) -> matplotlib.axes.Axes:
    if tprs_ci is not None:
        ax.fill_between(fprs, tprs_ci[0, :], tprs_ci[1, :], alpha=0.33, label='_nolegend_', color=color)
    ax.plot(fprs, tprs, color=color, linestyle=linestyle)

    return ax


# %%
def calc_frocs(
    froc_manifest_list: List[pd.DataFrame],
    mrn_list: List[str],
    legends: List[str] = None,
    seed: int = 0,
    plot_ci: bool = True,
    verbose: int = 1
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, List[dict]]:
    '''
    @froc_manifest_list: each manifest must have two columns: "prob", "label"
    @mrn_list: the full list of mrns, used to identify which patients have no label or predictions
    @seed: the random seed for ci calculation
    @plot_ci: if plot confidence interval or not. It will always be calculated.

    @return
    @ax: the axes object that contains the plot
    @froc_curves: a list of all the froc curve information (dictionary).
    '''

    rng = np.random.default_rng(seed)
    frocs = []
    if verbose > 0:
        print('Number of FROCS = {0}'.format(len(froc_manifest_list)))
    for i, manifest in enumerate(froc_manifest_list):
        mrn_empty = np.array([m for m in mrn_list if m not in manifest['MRN'].values])

        # fpr
        if verbose > 0:
            print('Plotting FROC {0}'.format(i), flush=True)
        froc = froc_with_ci(
            manifest['prob'].values,
            manifest['label'].values,
            manifest['MRN'].values,
            mrn_empty,
            seed=rng,
        )
        frocs.append(froc)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycle = ['-', '--']
    fig, ax = plt.subplots(figsize=[6, 4], dpi=100)
    for i in range(len(frocs)):
        color = colors[i // 2]
        linestyle = linestyle_cycle[i % 2]
        if plot_ci:
            ci = frocs[i]['tprs_ci']
        else:
            ci = None
        ax = plot_roc_with_ci(
            ax, frocs[i]['fpr_per_patient'], frocs[i]['tprs'], ci, color=color, linestyle=linestyle
        )

    ax.set_xscale('log')
    ax.set_xlabel('#FP/Image')
    ax.set_ylabel('Sensitivity')
    ax.grid(True)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([0, 1])

    if legends is not None:
        ax.legend([legends[i] for i in range(len(legends))])

    return fig, ax, frocs
