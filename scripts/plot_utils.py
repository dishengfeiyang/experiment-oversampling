"""共享绘图样式和工具函数"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def set_style():
    mpl.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
        'errorbar.capsize': 3,
    })

COLORS = {
    'theoretical':  '#C0392B',
    'real_only':    '#7F8C8D',
    'naive':        '#E67E22',
    'fixed_s':      '#2980B9',
    'theory_line':  '#27AE60',
    'shade':        '#ECF0F1',
    'highlight':    '#8E44AD',
}

MARKERS = {
    'theoretical': 'o',
    'real_only':   's',
    'naive':       '^',
    'fixed_s':     'D',
    'theory_line': None,
}

def savefig(fig, path, close=True):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    if close:
        plt.close(fig)
    print(f"[saved] {path}")