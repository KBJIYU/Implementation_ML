from tqdm import tqdm
import warnings
from pandas_profiling import ProfileReport
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import chartify
import seaborn as sns
import os
import sys
import time
import requests as rs
import numpy as np
import pandas as pd
import feather as ft
import datetime as dt

# viz - matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

# viz - others
sns.set()

# viz-interact

# tqdm

# pandas_profiling

warnings.filterwarnings('ignore')


def explore_distribution_1d(df,
                            bins='auto',
                            figWidth: float = 12,
                            figHeight: float = 6,
                            showDf: bool = False):
    targetOption = widgets.Dropdown(
        options=df.columns,
        value=df.columns[0],
        description="請選擇欄位: ",
        disabled=False,
    )

    def viz(targetOption):
        # 處理title
        titleN = f"Distribution of {targetOption}"
        # 顯示圖
        sns.set(rc={'figure.figsize': (figWidth, figHeight)})
        try:
            plt.hist(df[targetOption], bins=bins)
            plt.xlabel(targetOption)
            plt.ylabel('value')
            plt.title(titleN)
            plt.show()
        except Exception as e:
            print(e)

        # 顯示表格
        if showDf:
            display(df)

    go_viz = interactive(viz, targetOption=targetOption)
    display(go_viz)


def explore_corr(df,
                 c_positive=0.5, c_negative=-0.5, mode=None,
                 figWidth: float = 12,
                 figHeight: float = 12,):
    ''' mode '''
    dfCorr = df.corr().round(2)

    # Generate a mask for the upper triangle
    sns.set_theme(style="white")

    mask = np.triu(np.ones_like(dfCorr, dtype=bool))

    filteredDf = dfCorr[((dfCorr >= c_positive) | (dfCorr <= c_negative))]
    plt.figure(figsize=(figWidth, figHeight))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        filteredDf,
        annot=True,
        mask=mask,
        cmap=cmap,
        linecolor='gray',
        linewidths=.1,
        square=True, )
    plt.show()

    so = pd.DataFrame(dfCorr.unstack().sort_values(
        kind="quicksort")).reset_index()
    so.columns = ["col1", "col2", "c"]
    so = so.query("-1<c<1")

    return so
