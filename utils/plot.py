import random
import re

import matplotlib._color_data as mcd
import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt

overlap = tuple(name for name in mcd.CSS4_COLORS
                if "xkcd:" + name in mcd.XKCD_COLORS)

FIGURES_DIR = 'figures/'

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def gen_colors(cluster_len):
    colors = []
    while len(colors) != cluster_len:
        item = random.choice(overlap)
        if item not in colors:
            colors.append(item)
    return colors


def plot_data(df, colors):
    labels = list(df['labels'].unique())

    for label in labels:
        x_y = df.loc[df['labels'] == label, ['x', 'y']]
        plt.plot(list(x_y['x']), list(x_y['y']), 'ro', color=colors[label])

    for i, row in df.iterrows():
        plt.annotate(str(int(row['labels'])), (row['x'], row['y']))

    plt.axis([-0.5, 18, -0.5, 18])
    plt.title('Fake Generated Clusters for KNeighborsClassifier model')

    plt.savefig(FIGURES_DIR + 'Figure_data' + '.png')
    plt.show()


def plot_prediction(model, df, point: tuple, colors):
    labels = list(df['labels'].unique())

    for label in labels:
        x_y = df.loc[df['labels'] == label, ['x', 'y']]
        plt.plot(list(x_y['x']), list(x_y['y']), 'ro', color=colors[label])

    for i, row in df.iterrows():
        plt.annotate(str(int(row['labels'])), (row['x'], row['y']))

    plt.axis([-0.5, 18, -0.5, 18])
    x, y = point[0], point[1]

    plt.plot(x, y, 'ro', marker='*', markersize=20)

    pred = model.predict(np.array([[x, y]]))[0]

    plt.title(f'predict for {x},{y}   class is {pred}')

    plt.savefig(FIGURES_DIR + f'Figure_pred_{x}_{y}' + '.png')
    plt.show()


def plot_cm(cm, param):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    title = re.sub("[{}' ,()]", '', str(param))
    plt.title(title)
    plt.savefig(FIGURES_DIR + f'Figure_cm_{title}' + '.png')
    plt.show()


def plot_boundary(model, data, dot_colors, param):
    x_min, x_max = data['features'][:, 0].min(), data['features'][:, 0].max()
    y_min, y_max = data['features'][:, 1].min(), data['features'][:, 1].max()

    h = (x_max / x_min) / 100
    x_min -= h
    y_min -= h
    x_max += h
    y_max += h

    h = h / 8

    x_dots = np.arange(x_min, x_max, h)
    y_dots = np.arange(y_min, y_max, h)
    x_torow, y_tocol = np.meshgrid(x_dots, y_dots)
    x_flatten_rep = x_torow.ravel()
    y_flatten_rep = y_tocol.ravel()
    x_y = np.c_[x_flatten_rep, y_flatten_rep]
    preds = model.predict(x_y)
    preds = preds.reshape(x_torow.shape)

    cmap = colors.ListedColormap(dot_colors)

    plt.subplot(1, 1, 1)
    plt.contourf(x_torow, y_tocol, preds, cmap=cmap, alpha=0.8)
    plt.scatter(data['features'][:, 0], data['features'][:, 1], c=data['labels'], cmap=cmap)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    title = re.sub("[{}' ,()]", '', str(param))
    plt.title(f'svm {title}')
    plt.savefig(FIGURES_DIR + f'Figure_preds_{title}' + '.png')
    plt.show()
