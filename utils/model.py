import glob
import os
import random
import string

import joblib
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

from utils.plot import plot_cm

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """

    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def generate_model_name(size=5):
    """
    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def load_model():
    """
    :return: load latest modified weight model
    """
    path = latest_modified_weight()

    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def dump_model(model):
    model_name = WEIGHTS_DIR + 'svm-' + generate_model_name(5) + '.pkl'
    with open(model_name, 'wb') as f:
        joblib.dump(value=model, filename=f, compress=3)
        print(f'Model saved at {model_name}')


def train_model(data, args, gs=False):
    if gs:

        model = GridSearchCV(estimator=svm.SVC(), param_grid=args, cv=10)

        model.fit(data['train']['features'], data['train']['labels'])

        preds = model.predict(data['test']['features'])

        score = accuracy_score(data['test']['labels'], preds)

        param = model.best_params_
        param['GridSearchCV'] = 'yes'
        param['accuracy'] = f'{score:.2f}'

        cm = confusion_matrix(data['test']['labels'], preds)
        plot_cm(cm, param)

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model)

        return model, param

    else:
        model = svm.SVC(C=args.C if args.__contains__('C') else 1.0,
                        kernel=args.kernel if args.__contains__('kernel') else 'rbf',
                        degree=args.degree if args.__contains__('degree') else 3,
                        gamma=args.gamma if args.__contains__('gamma') else 'scale',
                        )

        model.fit(data['train']['features'], data['train']['labels'])

        preds = model.predict(data['test']['features'])

        score = accuracy_score(data['test']['labels'], preds)

        param = {}
        param['C'] = model.C,
        param['kernel'] = model.kernel,
        param['degree'] = model.degree,
        param['gamma'] = model.gamma,
        param['accuracy'] = f'{score:.2f}'

        cm = confusion_matrix(data['test']['labels'], preds)
        plot_cm(cm, param)

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model)

        return model, param
