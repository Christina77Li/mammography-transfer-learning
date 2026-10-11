"""Render experiment metrics and training curves."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#7b8798', '#3b86b6', '#c08745', '#2b806e']


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e9ed', linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=9)


def save(fig, directory, stem):
    fig.savefig(directory/f'{stem}.png', dpi=200, facecolor='white')
    svg = directory/f'{stem}.svg'
    fig.savefig(svg, facecolor='white', metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def comparison(results, directory):
    rows = read_rows(results/'reported_metrics.csv')
    validation = next(row for row in read_rows(results/'dataset_summary.csv') if row['partition'] == 'validation')
    chance_ap = int(validation['positive_images']) / int(validation['total_images'])
    labels = [row['experiment'] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, key, title, ylabel in zip(axes, ['roc_auc', 'pr_auc', 'val_loss'], ['ROC-AUC  ·  higher is better', 'PR-AUC (AP)  ·  higher is better', 'Validation loss  ·  lower is better'], ['ROC-AUC', 'Average precision', 'Validation loss']):
        values = [float(row[key]) for row in rows]
        ax.bar(labels, values, color=COLORS, width=.65)
        style(ax, ylabel)
        ax.set_ylim(0, max(values)*1.18)
        ax.set_title(title, fontsize=11, loc='left', pad=13)
        for i, value in enumerate(values):
            ax.text(i, value+max(values)*.025, f'{value:.4f}', ha='center', fontsize=10)
        if key == 'pr_auc':
            ax.axhline(chance_ap, color='#425166', linestyle='--', linewidth=1,
                       label=f'No-skill AP ≈ {chance_ap:.4f}')
            ax.legend(frameon=False, fontsize=8, loc='upper left')
        if key == 'val_loss':
            ax.set_ylabel('Weighted BCE (positive weight ≈ 6.76)')
    fig.suptitle('Reported validation results by training strategy', x=.07, ha='left', fontsize=16, fontweight='bold')
    fig.text(.07,.085,'Exp1: direct fine-tuning   |   Exp2b: warm-up → full data   |   Exp3: gradual unfreezing   |   Exp4: combined', fontsize=9, color='#425166')
    fig.text(.07,.035,'One checkpoint per experiment · Single-run validation results', fontsize=9, color='#5b6878')
    fig.subplots_adjust(left=.07, right=.98, top=.79, bottom=.23, wspace=.34)
    save(fig,directory,'strategy_comparison')


def stages(results, directory):
    rows = read_rows(results/'stage_checkpoints.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for col, (experiment, color) in enumerate([('Exp3',COLORS[2]),('Exp4',COLORS[3])]):
        records = [row for row in rows if row['experiment'] == experiment]
        labels = [('Head recalib.' if row['stage']=='recalibration' else row['stage'].capitalize())+f"\ne{row['global_epoch']}" for row in records]
        xs = np.arange(len(records))
        best_index = max(range(len(records)),key=lambda i:float(records[i]['roc_auc']))
        for ax,key,ylabel in [(axes[0,col],'roc_auc','ROC-AUC'),(axes[1,col],'val_loss','Validation loss')]:
            values = [float(row[key]) for row in records]
            ax.scatter(xs,values,color=color,s=42,zorder=3)
            ax.scatter([best_index],[values[best_index]],facecolors='none',edgecolors='#1b2533',s=150,linewidths=1.3,zorder=4)
            style(ax,ylabel)
            ax.set_xticks(xs,labels,fontsize=9)
            padding=(max(values)-min(values))*.35+.002
            ax.set_ylim(min(values)-padding,max(values)+padding)
            for x,v in zip(xs,values): ax.annotate(f'{v:.4f}',(x,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        axes[0,col].set_title(f'{experiment} · reported stage checkpoints',fontsize=12,loc='left',pad=16)
    fig.suptitle('Best ROC-AUC occurred before complete unfreezing',x=.07,ha='left',fontsize=16,fontweight='bold')
    fig.text(.07,.035,'One selected checkpoint per stage. Ring marks the highest ROC-AUC checkpoint.',fontsize=9,color='#5b6878')
    fig.subplots_adjust(left=.07,right=.98,top=.85,bottom=.13,wspace=.26,hspace=.44)
    save(fig,directory,'stage_checkpoints')


def baseline(results,directory):
    rows=read_rows(results/'exp1_history.csv')
    epochs=[int(row['epoch']) for row in rows]
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.4))
    axes[0].plot(epochs,[float(row['roc_auc']) for row in rows],color=COLORS[1],marker='o',markersize=4)
    selected=next(row for row in rows if int(row['epoch'])==9)
    axes[0].scatter([9],[float(selected['roc_auc'])],color=COLORS[3],s=70,zorder=5)
    axes[0].annotate('Reported checkpoint\nepoch 9 · 0.5780',(9,.578),xytext=(8,.555),fontsize=9,arrowprops={'arrowstyle':'-','color':'#607080'})
    axes[0].set_title('Validation ROC-AUC',loc='left',fontsize=11)
    style(axes[0],'ROC-AUC')
    for key,label,color in [('train_loss','Training',COLORS[0]),('val_loss','Validation',COLORS[1])]:
        axes[1].plot(epochs,[float(row[key]) for row in rows],color=color,marker='o',markersize=3,label=label)
    axes[1].legend(frameon=False,fontsize=9)
    axes[1].set_title('Logged training and validation loss',loc='left',fontsize=11)
    style(axes[1],'Loss')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_xticks([1,3,5,7,9,11,14])
    fig.suptitle('Exp1 · baseline training',x=.07,ha='left',fontsize=16,fontweight='bold')
    fig.text(.07,.045,'Epochs 1–14 · Validation recall at threshold 0.5 is zero throughout',fontsize=9,color='#5b6878')
    fig.subplots_adjust(left=.07,right=.98,top=.79,bottom=.22,wspace=.25)
    save(fig,directory,'exp1_training_curves')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,help='Defaults to <results>/figures.')
    args=parser.parse_args()
    directory=args.output or args.results/'figures'
    directory.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'mammography-finetuning','axes.labelcolor':'#263445','text.color':'#263445'}):
        comparison(args.results,directory)
        stages(args.results,directory)
        baseline(args.results,directory)
    print(f'Created 3 figures in PNG/SVG: {directory}')


if __name__=='__main__': main()
