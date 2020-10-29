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

sns.set()
warnings.filterwarnings('ignore')

# viz


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


# ml tools

def shape_sizeInfo(colX, colY, X, y, X_train, y_train, X_test, y_test):
    sizeInfo = {
        "column": dict(colX=colX, colY=colY),
        "full": dict(X_shape=X.shape, y_shape=y.shape),
        "train": dict(X_train=X_train.shape, y_train=y_train.shape),
        "test": dict(X_test=X_test.shape, y_test=y_test.shape)
    }

    return sizeInfo


# model - lr
def make_linearRegressionModel(df,
                               colX: str,
                               colY: str,
                               testSizeP: float = 1 / 3,
                               randomState: bool = 0):

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # split
    X = np.array(df[colX]).reshape(-1, 1)
    y = np.array(df[colY]).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSizeP, random_state=randomState)

    # size info
    sizeInfo = shape_sizeInfo(colX, colY, X, y, X_train, y_train, X_test,
                              y_test)

    # train
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # predict train
    predictInfo_train = predict_lr(model=regressor,
                                   x_data=X_train,
                                   y_data=y_train,
                                   mode='train')
    predictInfo_test = predict_lr(model=regressor,
                                  x_data=X_test,
                                  y_data=y_test,
                                  mode='test')

    lrInfo = dict(sizeInfo=sizeInfo,
                  predictInfo_train=predictInfo_train,
                  predictInfo_test=predictInfo_test)

    return lrInfo


def predict_lr(model, x_data, y_data, mode):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    y_pred = model.predict(x_data)
    rmse = (np.sqrt(mean_squared_error(y_data, y_pred)))

    r2 = r2_score(y_data, y_pred)  # r2 = model.score(x_data, y_data)

    return dict(mode=mode,
                model=model,
                x_data=x_data,
                y_data=y_data,
                y_pred=y_pred,
                rmse=rmse,
                r2=r2)


def viz_lr(lrInfo, mode):
    x_data = lrInfo[f'predictInfo_{mode}']['x_data']
    y_data = lrInfo[f'predictInfo_{mode}']['y_data']
    y_pred = lrInfo[f'predictInfo_{mode}']['y_pred']
    mode = lrInfo[f'predictInfo_{mode}']['mode']
    rmse = lrInfo[f'predictInfo_{mode}']['rmse']
    r2 = lrInfo[f'predictInfo_{mode}']['r2']
    colX = lrInfo['sizeInfo']['column']['colX']
    colY = lrInfo['sizeInfo']['column']['colY']

    print(f"{rmse=:,.4f}")
    print(f"{r2=:,.4f}")
    # visualizing the training set results
    plt.scatter(x_data, y_data, color='red')
    plt.plot(x_data, y_pred, color='blue')
    plt.title(f'Linear Regression - {mode} Data')
    plt.xlabel(colX)
    plt.ylabel(colY)
    plt.show()
