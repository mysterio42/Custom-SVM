import random

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from utils.plot import gen_colors

class CustomData:

    def __init__(self, path, nums):
        self.clusters = {}
        self.nums = nums
        self.path = path

    def _generate_dots(self, dots_range: tuple):
        return [random.uniform(dots_range[0], dots_range[1]) for _ in range(self.nums)]

    def generate_cluster(self, x_range: tuple, y_range: tuple, cluster_key: int):
        self.clusters[cluster_key] = {}

        self.clusters[cluster_key]['x'] = self._generate_dots(x_range)
        self.clusters[cluster_key]['y'] = self._generate_dots(y_range)

        self.clusters[cluster_key]['labels'] = [cluster_key for _ in range(self.nums)]

    @staticmethod
    def load_data(df):
        features = df[['x', 'y']].to_numpy()
        labels = df['labels'].to_numpy()

        data = train_test_split(features, labels, test_size=0.3, random_state=42)
        data = {
            'train': {
                'features': data[0],
                'labels': data[2],
            },
            'test': {
                'features': data[1],
                'labels': data[3]
            }
        }
        return data

    @staticmethod
    def generate_colors(df):
        num_clusters = len(list(df['labels'].unique()))
        return gen_colors(num_clusters)

    def to_json(self, name):
        data_name = self.path + name + '.json'
        with open(data_name, 'w') as f:
            json.dump(self.clusters, f)

    def to_csv(self, name):
        x = []
        y = []

        labels = []
        for el, item in enumerate(self.clusters.values()):
            x.extend(item['x'])
            y.extend(item['y'])
            labels.extend(item['labels'])

        data = {'x': x, 'y': y, 'labels': labels}
        data_name = self.path + name + '.csv'
        pd.DataFrame.from_dict(data).to_csv(data_name, index=False)
        del x,y,labels,data


