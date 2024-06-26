import pandas as pd
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import ast
import re

from pandas.core.frame import DataFrame
from sklearn.utils import resample
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from sklearn.metrics import f1_score
from .compare_auc_delong_xu import delong_roc_test


def _add_subplot_axes(ax, rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0] - .075
    y = infig_position[1] - .042
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height])  # ,axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def _show_values_on_bars(ax, values, cis, space=0.4, corr=0.1, fontsize=12, round_quant=False):
    first_pos = ax.patches[0].get_x() + ax.patches[0].get_width() + float(space) + 0.075
    for i, p in enumerate(ax.patches):
        _y = p.get_y() + p.get_height() + float(corr)
        # value = '{0:.3f}'.format(values[i])
        if round_quant:
            value = '{0:.2f}'.format(values[i])
            ci_str = ' $\pm$ {0:.2f}'.format(cis[i])
        else:
            value = '{0:.3f}'.format(values[i])
            ci_str = ' $\pm$ {0:.3f}'.format(cis[i])
        ax.text(first_pos, _y, str(value + ci_str), ha="left", fontsize=fontsize)


def _bootstrap_roc_95ci(df, n_iterations=2000):
    """ 
    Caclculate
    """

    n_size = df.shape[0] / 2
    # print(n_size)
    # run bootstrap
    aucs = list()
    for i in range(n_iterations):
        # prepare sample of df
        sample = resample(df, n_samples=n_size, stratify=df['y_true'])
        y_true = sample['y_true']
        y_score = sample['y_score']
        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)

    # mean
    m = np.mean(aucs)
    # standard error of mean/ standard deviation?
    se = np.std(aucs)
    # se=sem(aucs)
    width = 1.96 * se

    return width, m  # return width of CI and mean


def find_optimal_f1_threshold(y_true, y_scores, save_vals=False, dir_path_fname=None):
    if save_vals and dir_path_fname is not None:
        log_df = pd.DataFrame(columns=['threshold', 'f1', 'senstivity', 'specificity', 'sen_spec_diff'])
    else:
        log_df = None
    thresholds = np.linspace(0, 1, 100000)  # You can adjust the number of thresholds

    best_f1 = 0
    optimal_threshold = 0

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold

        if log_df is not None:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            sen_spec_diff = abs(sensitivity - specificity)
            log_df = log_df.append(
                {'threshold': threshold, 'f1': f1, 'senstivity': sensitivity, 'specificity': specificity,
                 'sen_spec_diff': sen_spec_diff}, ignore_index=True)
    # save the log_df
    if log_df is not None:
        log_df.to_csv(dir_path_fname)

    return optimal_threshold, best_f1


def plot_roc_with_auc_bars(df, events_to_compare=None, palette=None, savepth='', figname='fig_roc.png', title="",
                           show_legend=True, legend_fontsize=13, ci_shading=False, get_youdin=False,
                           get_f1_optimal=False, round_quant=False, grid=True):
    """
    Plots several mean ROC curves form a series of ROC data (can accept more than one fold of data for each prediction task).
    The plot has bars with bottom right corner reperenting mean AUC values with confidence intervals (which are calulated using 
    boostrapping with 1000 replicates by sns package). Colors can be adjusted by passinf in a snsn color palette, but the defalult 
    palette shoulf also look good. 
    
    Parameters:
    ---------------
    df - pandas dataframe containing following columns: 'task_name', 'fold', and 'y_score' and 'y_true'. 
    If there are more columns - no problem.
    'task_name'  - name for a specific ROC curve, e.g. 'Prediction by XgBoost' and 'Preciction by Logistic Regression'
    'fold' - If data comes from cross- validation each fold should be concatenaded to the same columns, just with a different fold no.
            Please refer do the aim_func.structurs.predictions for a useful helper class to concatenate predictions into apropriate df
            durung CV
    
    'events_to_compare' -  often we may have generated more types of predictions than we want to display on a single plot so then it
                            is possible to specify the predictions to display with a list

    Returns:
    ---------------
    None
    """
    font = font_manager.FontProperties(family='Arial',
                                       style='normal',
                                       size=legend_fontsize)
    margin = 0.02
    if (palette is None):
        palette = sns.color_palette("Set1", 25)

    if (events_to_compare is None):
        events_to_compare = df['task_name'].unique()  # then we use all unique event names
    if len(events_to_compare) > len(df['task_name'].unique()):
        raise ValueError(f"events to compare has extra cases: {events_to_compare} vs. {df['task_name'].unique()}")

    df = df[df['task_name'].isin(events_to_compare)]

    no_roc = df['task_name'].nunique()
    print(no_roc)
    if len(palette) < no_roc:
        raise ValueError("not enough colors in the pallette")
    else:
        colors = palette[:no_roc]

    fig = plt.figure(figsize=(10, 10))
    ax_roc = fig.add_subplot(1, 1, 1)
    # plt.title(task)
    # if legend_fontsize > 13:
    #     diff = legend_fontsize - 11
    # else:
    #     diff = 0
    # x, y, width, height
    diff = 12
    rect = [0.71 - 0.01 * (diff), 0.07, 0.2, 0.3]
    # use this code to move the legend
    # diff=-6
    # rect = [0.71-0.01*(diff),0.05,0.1,0.3]
    if show_legend:
        ax_bars = _add_subplot_axes(ax_roc, rect)

    # first calculate actual AUCs fora each complete dataset (all folds)
    auc_df = pd.DataFrame(columns=['task', 'auc'])
    for task in events_to_compare:
        all_roc_data = df[df['task_name'] == task]
        y_true = all_roc_data['y_true'].astype(float).tolist()
        y_score = all_roc_data['y_score'].astype(float).tolist()
        rocauc = roc_auc_score(y_true, y_score)

        width_of_ci, mean_from_bootstrap = _bootstrap_roc_95ci(all_roc_data, n_iterations=1000)

        auc_df = auc_df.append({'task': task,
                                'auc': rocauc,
                                'ci': width_of_ci,
                                'bootstrap_mean': mean_from_bootstrap,
                                'color': None}, ignore_index=True)

    # than get the right order:
    sort_order = auc_df.groupby('task').mean().sort_values('auc', ascending=False)
    sort_order = list(sort_order.index)
    # print("ORDER",sort_order)

    for i, task in enumerate(sort_order):
        task_idx = auc_df[auc_df['task'] == task].index[0]
        auc_df.loc[task_idx, 'color'] = colors[i]

    i = 0
    ax_roc.axis("square")
    ax_roc.set_xlim([0 - margin, 1. + margin])
    ax_roc.set_ylim([0 - margin, 1. + margin])
    ax_roc.tick_params(axis='both', which='major', labelsize=16)
    if title != "":
        ax_roc.set_title(title, fontproperties=font)
    ax_roc.set_xlabel("1 - Specificity", fontproperties=font)
    ax_roc.set_ylabel("Sensitivity", fontproperties=font)
    ax_roc.grid(grid)
    # ax_roc.plot([0, 1], [0, 1], ":k",)
    ax_roc.plot([0, 1], [0, 1], ":", color='lightgrey')
    # print(auc_df)

    if get_youdin:
        fpr_dict = dict()
        tpr_dict = dict()
        thresholds_dict = dict()

    for i, event in enumerate(events_to_compare):
        # k=len(events_to_compare)-i-1
        # task=sort_order[k]

        all_roc_data = df[df['task_name'] == event]
        y_true = all_roc_data['y_true'].astype(float).tolist()
        y_score = all_roc_data['y_score'].astype(float).tolist()

        fpr, tpr, tr = roc_curve(y_true, y_score)

        ax_roc.plot(fpr, tpr, linewidth=3.5, color=auc_df[auc_df['task'] == event]['color'].tolist()[0])  # , ax=ax_roc)
        if ci_shading:
            ci = auc_df[auc_df['task'] == event]['ci'].tolist()[0]
            ax_roc.fill_between(fpr, (tpr - ci), (tpr + ci), color=auc_df[auc_df['task'] == event]['color'].tolist()[0],
                                alpha=.1)
        if get_youdin:
            fpr_dict[event] = fpr
            tpr_dict[event] = tpr
            thresholds_dict[event] = tr
        i = i + 1
    # print(fpr_dict.keys())
    # print(tpr_dict.keys())
    # print(thresholds_dict.keys())
    if get_youdin:
        J_stats = [None] * len(events_to_compare)
        opt_thresholds = [None] * len(events_to_compare)
        for i, event in enumerate(events_to_compare):
            J_stats[i] = tpr_dict[event] - fpr_dict[event]
            opt_thresholds[i] = thresholds_dict[event][np.argmax(J_stats[i])]
            print(f'Optimum Youdens threshold for {event}: {str(opt_thresholds[i])}')

    if get_f1_optimal:
        for i, event in enumerate(events_to_compare):
            all_roc_data = df[df['task_name'] == event]
            y_true = all_roc_data['y_true'].astype(float).tolist()
            y_score = all_roc_data['y_score'].astype(float).tolist()
            thresh, f1 = find_optimal_f1_threshold(y_true, y_score, save_vals=True,
                                                   dir_path_fname=savepth + f'f1_threshold_log_{event}.csv')
            print(f'Optimum F1 threshold for {event}: {thresh}, (score: {f1})')

    # ax_roc.get_legend().remove()

    # print(sort_order)
    aucs = list(auc_df.groupby('task').mean().sort_values('auc', ascending=False).to_numpy()[:, 0])
    if round_quant: aucs = [round(num, 2) for num in aucs]

    if show_legend:
        auc_df_legend = auc_df.copy()
        auc_df_legend['auc'] = auc_df_legend['auc'] - 0.0
        sns.barplot(x='auc', y='task', data=auc_df_legend, order=sort_order, palette=colors, ax=ax_bars)
        # sns.barplot(x='auc',y='task',data=auc_df_legend,palette=colors,ax=ax_bars)

    dummy_df = pd.Series(sort_order, name='task').to_frame()
    sorted_df = pd.merge(dummy_df, auc_df, on='task', how='left')

    cis = list(sorted_df['ci'])
    if round_quant: cis = [round(num, 2) for num in cis]

    if show_legend:
        ax_bars.errorbar(y=range(len(sorted_df)),
                         x=sorted_df['auc'] - 0.0,
                         xerr=sorted_df['ci'],
                         fmt='none',
                         linewidth=2,
                         c='black',
                         capsize=5)

        for tick in ax_bars.yaxis.get_major_ticks():
            # tick.label.set_fontsize(legend_fontsize)
            tick.label.set_fontproperties(font)
        ax_bars.set_ylabel('')
        ax_bars.xaxis.set_visible(False)
        _show_values_on_bars(ax_bars, aucs, cis, 0.1, corr=-0.1, fontsize=legend_fontsize, round_quant=round_quant)
        ax_bars.set(frame_on=False)
    plt.savefig(os.path.join(savepth, figname))
    plt.show()

    auc_df = auc_df.sort_values(by='auc', ascending=False).reset_index()
    ### adding Delong Results to DataFRame
    # rint(auc_df)
    name_best = auc_df.iloc[0, 1]
    # print("Name best ", name_best)
    best_scores = df[df['task_name'] == name_best].sort_values(by='index')
    # print(best_scores)
    best_scores = best_scores['y_score']

    def _get_vals(task, valname):
        # scores=df[df['task_name']==task]
        scores = df[df['task_name'] == task].sort_values(by='index')
        # print(task, valname)
        # print(scores)
        return scores[valname]

    print(auc_df)
    # print(best_scores)
    auc_df.loc[1:, "deLong"] = auc_df.loc[1:, :].apply(
        lambda row: '{0:.15f}'.format(delong_roc_test(_get_vals(row['task'], 'y_true'),
                                                      best_scores,
                                                      _get_vals(row['task'], 'y_score'))[0][0]), axis=1).values
    to_return = auc_df.drop(columns=['bootstrap_mean'])

    return to_return
    # return None


# Inspo from: https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
def plot_roc_with_auc_bars_multiclass_OVR(df, events_to_compare=None, classes=[], palette=None, savepth='',
                                          figname='fig_roc.png', title="", show_legend=True, legend_fontsize=13,
                                          ci_shading=False, save_path=None, get_youdin=True, round_quant=False):
    if len(classes) < 2:
        raise ValueError

    margin = 0.02
    if (palette is None):
        palette = sns.color_palette("Set1", 25)

    if (events_to_compare is None):
        events_to_compare = df['task_name'].unique()  # then we use all unique event names

    df = df[df['task_name'].isin(events_to_compare)]

    no_roc = df['task_name'].nunique()
    print(no_roc)
    if len(palette) < no_roc:
        raise ValueError("not enough colors in the pallette")
    else:
        colors = palette[:no_roc]

    fig = plt.figure(figsize=(30, 15))

    # plt.title(task)
    # if legend_fontsize > 13:
    #     diff = legend_fontsize - 11
    # else:
    #     diff = 0
    diff = 2
    # x, y, width, height
    # rect = [0.71-0.01*(diff),0.07,0.2,0.2]
    # if show_legend:
    #     ax_bars = _add_subplot_axes(ax_roc,rect)

    for i in range(len(classes)):
        c = classes[i]
        auc_df = pd.DataFrame(columns=['task', 'auc'])
        # some plot config
        ax_cls = fig.add_subplot(1, len(classes), i + 1)
        ax_cls.axis("square")
        ax_cls.set_xlim([0 - margin, 1. + margin])
        ax_cls.set_ylim([0 - margin, 1. + margin])
        ax_cls.tick_params(axis='both', which='major', labelsize=16)
        # if title != "":
        ax_cls.set_title(f"{c} One vs. Rest AUC", fontsize=16)
        ax_cls.set_xlabel("1 - specificity", fontsize=16)
        ax_cls.set_ylabel("sensitivity", fontsize=16)
        ax_cls.grid(True)
        # ax_roc.plot([0, 1], [0, 1], ":k",)
        ax_cls.plot([0, 1], [0, 1], ":", color='lightgrey')

        if show_legend:
            rect = [0.85 - 0.01 * (diff), 0.1, 0.2, 0.2]
            ax_bars = _add_subplot_axes(ax_cls, rect)

        for k, task in enumerate(events_to_compare):
            all_roc_data = df[df['task_name'] == task]
            y_true = [1 if y == i else 0 for y in all_roc_data['y_true']]
            all_roc_data['y_true'] = y_true
            y_scores = all_roc_data['y_score']
            y_score = [float(ast.literal_eval(re.sub(r'\s+', ", ", x))[i]) for x in y_scores]
            all_roc_data['y_score'] = y_score
            roc_auc = roc_auc_score(y_true, y_score)

            width_of_ci, mean_from_bootstrap = _bootstrap_roc_95ci(all_roc_data, n_iterations=1000)
            auc_df = auc_df.append({'task': task,
                                    'auc': roc_auc,
                                    'ci': width_of_ci,
                                    'bootstrap_mean': mean_from_bootstrap,
                                    'color': None}, ignore_index=True)
            sort_order = auc_df.groupby('task').mean().sort_values('auc', ascending=False)
            sort_order = list(sort_order.index)

            for j, task in enumerate(sort_order):
                task_idx = auc_df[auc_df['task'] == task].index[0]
                auc_df.loc[task_idx, 'color'] = colors[j]

            # print(auc_df)

            # if get_youdin:
            #     fpr_dict  = dict()
            #     tpr_dict = dict()
            #     thresholds_dict = dict()

            fpr, tpr, tr = roc_curve(y_true, y_score)

            ax_cls.plot(fpr, tpr, linewidth=3.5,
                        color=auc_df[auc_df['task'] == task]['color'].tolist()[0])  # , ax=ax_roc)
            if ci_shading:
                ci = auc_df[auc_df['task'] == task]['ci'].tolist()[0]
                ax_cls.fill_between(fpr, (tpr - ci), (tpr + ci),
                                    color=auc_df[auc_df['task'] == event]['color'].tolist()[0], alpha=.1)

        aucs = list(auc_df.groupby('task').mean().sort_values('auc', ascending=False).to_numpy()[:, 0])
        if round_quant: aucs = [round(num, 2) for num in aucs]

        if show_legend:
            auc_df_legend = auc_df.copy()
            auc_df_legend['auc'] = auc_df_legend['auc'] - 0.0
            sns.barplot(x='auc', y='task', data=auc_df_legend, order=sort_order, palette=colors, ax=ax_bars)

        dummy_df = pd.Series(sort_order, name='task').to_frame()
        sorted_df = pd.merge(dummy_df, auc_df, on='task', how='left')

        cis = list(sorted_df['ci'])
        if round_quant: cis = [round(num, 2) for num in cis]

        if show_legend:
            ax_bars.errorbar(y=range(len(sorted_df)),
                             x=sorted_df['auc'] - 0.0,
                             xerr=sorted_df['ci'],
                             fmt='none',
                             linewidth=2,
                             c='black',
                             capsize=5)

            for tick in ax_bars.yaxis.get_major_ticks():
                tick.label.set_fontsize(legend_fontsize)
            ax_bars.set_ylabel('')
            ax_bars.xaxis.set_visible(False)
            _show_values_on_bars(ax_bars, aucs, cis, 0.1, corr=-0.3, fontsize=legend_fontsize, round_quant=round_quant)
            ax_bars.set(frame_on=False)

        if len(events_to_compare) > 1:
            delongs_df = df.copy()
            y_true = [1 if y == i else 0 for y in delongs_df['y_true']]
            delongs_df['y_true'] = y_true
            y_scores = delongs_df['y_score']
            y_score = [float(ast.literal_eval(re.sub(r'\s+', ", ", x))[i]) for x in y_scores]
            delongs_df['y_score'] = y_score
            dl_df = complete_delongs(delongs_df)
            print(f"Delongs test for: {c}")
            print(dl_df)

    plt.savefig(os.path.join(savepth, figname))
    plt.show()

    return None


# Inspo from: https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
def plot_roc_with_auc_bars_multiclass_OVO(df, events_to_compare=None, classes=[], palette=None, savepth='',
                                          figname='fig_roc.png', title="", show_legend=True, legend_fontsize=13,
                                          ci_shading=False, save_path=None, get_youdin=True, round_quant=False):
    if len(classes) < 2:
        raise ValueError

    margin = 0.02
    if (palette is None):
        palette = sns.color_palette("Set1", 25)

    if (events_to_compare is None):
        events_to_compare = df['task_name'].unique()  # then we use all unique event names

    df = df[df['task_name'].isin(events_to_compare)]

    no_roc = df['task_name'].nunique()
    print(no_roc)
    if len(palette) < no_roc:
        raise ValueError("not enough colors in the pallette")
    else:
        colors = palette[:no_roc]

    fig = plt.figure(figsize=(60, 15))

    diff = 2

    class_combos = []
    for c1 in range(len(classes)):
        for c2 in range(c1 + 1, len(classes)):
            class_combos.append([c1, c2])
            class_combos.append([c2, c1])

    print(f"Number of OVO permutations: {len(class_combos)}")

    for i in range(len(class_combos)):
        combo = class_combos[i]
        c1 = combo[0]
        c2 = combo[1]
        # c1_index = classes.index(c1)
        title = f"{classes[c1]} vs. {classes[c2]}: AUC"

        # c = classes[i]
        auc_df = pd.DataFrame(columns=['task', 'auc'])
        # some plot config
        ax_cls = fig.add_subplot(1, len(class_combos), i + 1)
        ax_cls.axis("square")
        ax_cls.set_xlim([0 - margin, 1. + margin])
        ax_cls.set_ylim([0 - margin, 1. + margin])
        ax_cls.tick_params(axis='both', which='major', labelsize=16)
        # if title != "":
        ax_cls.set_title(title, fontsize=16)
        ax_cls.set_xlabel("1 - specificity", fontsize=16)
        ax_cls.set_ylabel("sensitivity", fontsize=16)
        ax_cls.grid(True)
        # ax_roc.plot([0, 1], [0, 1], ":k",)
        ax_cls.plot([0, 1], [0, 1], ":", color='lightgrey')

        if show_legend:
            rect = [1.2 - 0.01 * (diff), 0.1, 0.2, 0.2]
            ax_bars = _add_subplot_axes(ax_cls, rect)

        for k, task in enumerate(events_to_compare):
            all_roc_data = df[df['task_name'] == task]
            all_roc_data = all_roc_data[(all_roc_data['y_true'] == c1) | (all_roc_data['y_true'] == c2)]
            y_true = [1 if y == c1 else 0 for y in all_roc_data['y_true']]
            all_roc_data['y_true'] = y_true
            y_scores = all_roc_data['y_score']
            y_score = [float(ast.literal_eval(re.sub(r'\s+', ", ", x))[c1]) for x in y_scores]
            all_roc_data['y_score'] = y_score
            roc_auc = roc_auc_score(y_true, y_score)

            width_of_ci, mean_from_bootstrap = _bootstrap_roc_95ci(all_roc_data, n_iterations=1000)
            auc_df = auc_df.append({'task': task,
                                    'auc': roc_auc,
                                    'ci': width_of_ci,
                                    'bootstrap_mean': mean_from_bootstrap,
                                    'color': None}, ignore_index=True)
            sort_order = auc_df.groupby('task').mean().sort_values('auc', ascending=False)
            sort_order = list(sort_order.index)

            for j, task in enumerate(sort_order):
                task_idx = auc_df[auc_df['task'] == task].index[0]
                auc_df.loc[task_idx, 'color'] = colors[j]


            fpr, tpr, tr = roc_curve(y_true, y_score)

            ax_cls.plot(fpr, tpr, linewidth=3.5,
                        color=auc_df[auc_df['task'] == task]['color'].tolist()[0])  # , ax=ax_roc)
            if ci_shading:
                ci = auc_df[auc_df['task'] == task]['ci'].tolist()[0]
                ax_cls.fill_between(fpr, (tpr - ci), (tpr + ci),
                                    color=auc_df[auc_df['task'] == event]['color'].tolist()[0], alpha=.1)

        aucs = list(auc_df.groupby('task').mean().sort_values('auc', ascending=False).to_numpy()[:, 0])
        if round_quant: aucs = [round(num, 2) for num in aucs]

        if show_legend:
            auc_df_legend = auc_df.copy()
            auc_df_legend['auc'] = auc_df_legend['auc'] - 0.0
            sns.barplot(x='auc', y='task', data=auc_df_legend, order=sort_order, palette=colors, ax=ax_bars)

        dummy_df = pd.Series(sort_order, name='task').to_frame()
        sorted_df = pd.merge(dummy_df, auc_df, on='task', how='left')

        cis = list(sorted_df['ci'])
        if round_quant: cis = [round(num, 2) for num in cis]

        if show_legend:
            ax_bars.errorbar(y=range(len(sorted_df)),
                             x=sorted_df['auc'] - 0.0,
                             xerr=sorted_df['ci'],
                             fmt='none',
                             linewidth=2,
                             c='black',
                             capsize=5)

            for tick in ax_bars.yaxis.get_major_ticks():
                tick.label.set_fontsize(legend_fontsize)
            ax_bars.set_ylabel('')
            ax_bars.xaxis.set_visible(False)
            _show_values_on_bars(ax_bars, aucs, cis, 0.1, corr=-0.3, fontsize=legend_fontsize, round_quant=round_quant)
            ax_bars.set(frame_on=False)

        if len(events_to_compare) > 1:
            delongs_df = df.copy()
            delongs_df = delongs_df[(delongs_df['y_true'] == c1) | (delongs_df['y_true'] == c2)]
            y_true = [1 if y == c1 else 0 for y in delongs_df['y_true']]
            delongs_df['y_true'] = y_true
            y_scores = delongs_df['y_score']
            y_score = [float(ast.literal_eval(re.sub(r'\s+', ", ", x))[c1]) for x in y_scores]
            delongs_df['y_score'] = y_score
            dl_df = complete_delongs(delongs_df)
            print(f"Delongs test for: {c1} vs. {c2}")
            print(dl_df)

    plt.savefig(os.path.join(savepth, figname))
    plt.show()

    return None


def plot_roc_auc_kp(df, prediction_labels, y_true_label='y_true', palette=["#8AC926", "#1982C4", "#FFCA3A", "#FF595E"],
                    pub_labels=None, savepth='', figname='fig_roc.svg', bootstrap=True, auc_lim=.5, title=None):
    """
    Plots several mean ROC curves wit auc bar plot
    
    Parameters:
    ---------------
    df - pandas dataframe containing 3 of more colums, one of which is y_true or can have a name specified by y_true_label 
   
    prediction_labels - names of columns in the df that specify the scores to be compared

    pub_labels - nicer names to be dispayed on plot

    auc_lim - the beggining of x_axis for bar plot

    palette - colors will be assigned to labels in this order

   

    Returns:
    ---------------
    None
    """

    margin = 0.02

    if (palette is None):
        palette = sns.color_palette("Set1", 25)
    elif (type(palette) == 'list'):
        palette = sns.color_palette(palette)

    no_roc = len(prediction_labels)
    colors = palette[:no_roc]

    fig = plt.figure(figsize=(10, 10))

    ax_roc = fig.add_subplot(1, 1, 1)
    if (title):
        ax_roc.set_title(title, fontsize=20)
    # plt.title(task)
    rect = [0.7, 0.07, 0.2, 0.2]
    ax_bars = _add_subplot_axes(ax_roc, rect)

    # first calculate actual AUCs fora each complete dataset (all folds)
    auc_df = pd.DataFrame(columns=['label', 'auc', 'color'])
    for i, label in enumerate(prediction_labels):

        y_true = df[y_true_label].astype(float).tolist()
        y_score = df[label].astype(float).tolist()

        all_roc_data = pd.DataFrame({'y_score': y_score, 'y_true': y_true})  # for bootstrapping

        color = colors[i]  # so that color is assigned to the label

        rocauc = roc_auc_score(y_true, y_score)  # calculate actual AUC

        # plot AUC curves starting from the greatest one

        fpr, tpr, tr = roc_curve(y_true, y_score)
        # roc=compute_roc(y_score,y_true,1)
        # plot_roc(roc, color=colors[i])
        ax_roc.axis("square")
        ax_roc.set_xlim([0 - margin, 1. + margin])
        ax_roc.set_ylim([0 - margin, 1. + margin])
        ax_roc.tick_params(axis='both', which='major', labelsize=20)

        ax_roc.set_xlabel("FPR (false positive rate)", fontsize=20)
        ax_roc.set_ylabel("TPR (true positive rate)", fontsize=20)
        ax_roc.grid(True)
        ax_roc.plot([0, 1], [0, 1], ":k")
        ax_roc.plot(fpr, tpr, linewidth=2, color=color)  # , ax=ax_roc)

        # calculate bootstrappitn CIs

        if (bootstrap):
            width_of_ci, mean_from_bootstrap = _bootstrap_roc_95ci(all_roc_data,
                                                                   n_iterations=1000)  # calculate cis and mean
        else:
            width_of_ci = None
            mean_from_bootstrap = None

        auc_df = auc_df.append({'label': label,
                                'auc': rocauc,
                                'ci': width_of_ci,
                                'bootstrap_mean': mean_from_bootstrap, 'color': color}, ignore_index=True
                               )
        # auc_df=auc_df.reset_index()

    # than geet the right order, from gratest to lowest, for correct print out of aucs:
    sorted_auc_df = auc_df.sort_values('auc', ascending=False)

    sort_order = list(sorted_auc_df.index)
    # print(sort_order)
    # sort_order=[1,2]
    aucs = list(auc_df.groupby('label').mean().sort_values('auc', ascending=False).to_numpy()[:,
                0])  # actual auc values, from gretest to lowest

    # print(auc_df)
    # print(aucs)

    ordered_colors = [colors[i] for i in sort_order]
    # print(auc_df)
    sns.barplot(x='auc', y='label', data=auc_df, order=sorted_auc_df['label'], palette=ordered_colors, ax=ax_bars)

    ax_bars.set_xlim(auc_lim)

    if (bootstrap):  # adding errorbars
        dummy_df = pd.Series(sorted_auc_df['label'], name='label').to_frame()

        sorted_df = pd.merge(dummy_df, auc_df, on='label', how='left')
        ax_bars.errorbar(y=range(len(sorted_df)),
                         x=sorted_df['auc'],
                         xerr=sorted_df['ci'],
                         fmt='none',
                         linewidth=2, c='black',
                         capsize=3)

        for tick in ax_bars.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        ax_bars.set_ylabel('')
        ax_bars.xaxis.set_visible(False)
        _show_values_on_bars(ax_bars, aucs, 0.05, corr=-0.2, fontsize=13)

    ax_bars.set(frame_on=False)
    plt.savefig(os.path.join(savepth, figname), dpi=2000)
    plt.show()

    auc_df = auc_df.sort_values(by='auc', ascending=False)  # .reset_index()
    ### adding Delong Results to DataFRame
    # rint(auc_df)
    name_best = auc_df.iloc[0, 0]
    # print("Name best ", name_best)
    best_scores = df[name_best]

    return None


"""
    def get_p_values(rocs, best_roc):
        for roc in rocs:
            if (roc.auc!=best_roc.auc):df[df['task_name']==row['task']]['y_true']
                
"""

if __name__ == "__main__":
    df = DataFrame({"a": [.4, .5, .8, .2, .8, .2, .3, .4],
                    "b": [.8, .2, .9, .1, .9, .8, .7, .5],
                    "y_true": [0, 0, 1, 0, 1, 1, 0, 1]})

    plot_roc_auc(df, ["a", "b"], bootstrap=True)

    """
    
    # for debugging etc
    df=pd.read_csv('/home/konrad/aim/dev/scripts/aim_func/example_data/preds LR vs XGB.csv')
    aucs=plot_roc_with_auc_bars(df, ['XgBoost','LR'])

   # print(aucs.iloc[1,3])
    #print(type(aucs.iloc[1,3]))
   # print('{0:.10f}'.format(aucs.iloc[1,3]))
    print(aucs)"""


def complete_delongs(preds_df, events_to_compare=None, verbose=0):
    # get sorted list of auc performances
    if (events_to_compare is None):
        events_to_compare = preds_df['task_name'].unique()  # then we use all unique event names
    preds_df = preds_df[preds_df['task_name'].isin(events_to_compare)]

    # first calculate actual AUCs fora each complete dataset (all folds)
    auc_df = pd.DataFrame(columns=['task', 'auc'])
    for task in events_to_compare:
        all_roc_data = preds_df[preds_df['task_name'] == task]
        y_true = all_roc_data['y_true'].astype(float).tolist()
        y_score = all_roc_data['y_score'].astype(float).tolist()
        rocauc = roc_auc_score(y_true, y_score)

        width_of_ci, mean_from_bootstrap = _bootstrap_roc_95ci(all_roc_data, n_iterations=1000)

        auc_df = auc_df.append({'task': task,
                                'auc': rocauc,
                                'ci': width_of_ci,
                                'bootstrap_mean': mean_from_bootstrap}, ignore_index=True)

    # then get the right order:
    sort_order = auc_df.groupby('task').mean().sort_values('auc', ascending=False)
    sort_order = list(sort_order.index)
    auc_df = auc_df.sort_values(by='auc', ascending=False).reset_index()

    delongs_df = pd.DataFrame(index=sort_order, columns=sort_order)

    def _get_vals(task, valname):
        scores = preds_df[preds_df['task_name'] == task].sort_values(by='index')
        return scores[valname]

    # print("Sort order: ", sort_order)
    # print("short order cut: ", sort_order[:-1])
    for i, task in enumerate(sort_order[:-1]):
        name_best = auc_df.iloc[0 + i, 1]
        best_scores = preds_df[preds_df['task_name'] == name_best].sort_values(by='index')
        best_scores = best_scores['y_score']
        delongs_df.iloc[1 + i:, i] = auc_df.loc[1 + i:, :].apply(
            lambda row: '{0:.15f}'.format(delong_roc_test(_get_vals(row['task'], 'y_true'),
                                                          best_scores,
                                                          _get_vals(row['task'], 'y_score'))[0][0]), axis=1).values
    return delongs_df


def delong_test(y_true1, y_score1, y_true2, y_score2):
    n1 = len(y_true1)
    n2 = len(y_true2)
    auc1 = roc_auc(y_true1, y_score1)
    auc2 = roc_auc(y_true2, y_score2)

    u_stat = (n1 * n2) / 2
    u = u_stat - auc1 * n2
    v = auc2 * n1 + auc1 * n2 - u_stat

    # Calculate the covariance between AUC estimates
    cov_auc = (auc1 * (1 - auc1) + auc2 * (1 - auc2) - np.cov([y_score1, y_score2])[0, 1]) / n1 / n2

    # Calculate Z-score
    z = (u - 0.5) / np.sqrt(cov_auc)

    # Calculate p-value using normal distribution
    p_value = 2 * (1 - norm.cdf(np.abs(z)))

    return p_value


def delong_test_custom(y_true1, y_pred1, y_true2, y_pred2):
    n1 = len(y_true1)
    n2 = len(y_true2)

    auc1 = roc_auc_score(y_true1, y_pred1)
    auc2 = roc_auc_score(y_true2, y_pred2)

    z = (auc1 - auc2)

    var = (auc1 * (1 - auc1) + auc2 * (1 - auc2) + (auc1 - auc2) ** 2) / (n1 + n2 - 3)

    z_stat = z / np.sqrt(var)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return p_value


def complete_delongs_unpaired(preds_df, events_to_compare=None, verbose=0):
    # get sorted list of auc performances
    if (events_to_compare is None):
        events_to_compare = preds_df['task_name'].unique()  # then we use all unique event names
    preds_df = preds_df[preds_df['task_name'].isin(events_to_compare)]

    # first calculate actual AUCs fora each complete dataset (all folds)
    auc_df = pd.DataFrame(columns=['task', 'auc'])
    for task in events_to_compare:
        all_roc_data = preds_df[preds_df['task_name'] == task]
        y_true = all_roc_data['y_true'].astype(float).tolist()
        y_score = all_roc_data['y_score'].astype(float).tolist()
        rocauc = roc_auc_score(y_true, y_score)

        width_of_ci, mean_from_bootstrap = _bootstrap_roc_95ci(all_roc_data, n_iterations=1000)

        auc_df = auc_df.append({'task': task,
                                'auc': rocauc,
                                'ci': width_of_ci,
                                'bootstrap_mean': mean_from_bootstrap}, ignore_index=True)

    # then get the right order:
    sort_order = auc_df.groupby('task').mean().sort_values('auc', ascending=False)
    sort_order = list(sort_order.index)
    auc_df = auc_df.sort_values(by='auc', ascending=False).reset_index()

    delongs_df = pd.DataFrame(index=sort_order, columns=sort_order)

    def _get_vals(task, valname):
        scores = preds_df[preds_df['task_name'] == task].sort_values(by='index')
        return scores[valname]

    # print("Sort order: ", sort_order)
    # print("short order cut: ", sort_order[:-1])
    for i, task in enumerate(sort_order[:-1]):
        name_best = auc_df.iloc[0 + i, 1]
        best_scores_df = preds_df[preds_df['task_name'] == name_best].sort_values(by='index')
        best_scores = best_scores_df['y_score']
        best_true = best_scores_df['y_true']
        delongs_df.iloc[1 + i:, i] = auc_df.loc[1 + i:, :].apply(lambda row: '{0:.15f}'.format(
            delong_test_custom(best_true, best_scores, _get_vals(row['task'], 'y_true'),
                               _get_vals(row['task'], 'y_score'))), axis=1).values
    return delongs_df
