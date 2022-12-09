import pandas as pd
from sklearn.model_selection import StratifiedKFold

PATH_NORMAL = 'dataset/normal'
PATH_PNEUMONIA = 'dataset/pneumonia'
PATH_TUBERCULOSIS = 'dataset/tuberculosis'

NORMAL = 0
PNEUMONIA = 1
TUBERCULOSIS = 2


def make_list_of_patients():
    data = pd.read_csv('dataset/labels_train.csv', sep=',')

    patients = pd.DataFrame(columns=['ID', 'paths', 'label'])

    for _, row in data.iterrows():
        patient_ID = row['file'].split('_')[0]
        if len(patients.loc[patients['ID'] == patient_ID, 'paths']) == 0:
            new_patient = {'ID': patient_ID, 'paths': row['file'], 'label': row['label']}
            patients = patients.append(new_patient, ignore_index=True)
        else:
            old_label = patients.loc[patients['ID'] == patient_ID, 'label']
            assert old_label.values[0] == row['label']  # make sure two samples with the same ID do not have
            # different diagnosis
            paths = patients.loc[patients['ID'] == patient_ID, 'paths'].to_list()
            paths.append(row['file'])
            updated_patient = {'ID': patient_ID, 'paths': paths, 'label': row['label']}
            patients.drop(patients[patients.ID == patient_ID].index, inplace=True)
            patients = patients.append(updated_patient, ignore_index=True)

    patients = patients.replace('N', NORMAL)
    patients = patients.replace('P', PNEUMONIA)
    patients = patients.replace('T', TUBERCULOSIS)
    return patients


def cross_validation_splits(data: pd.DataFrame, fold=5, shuffle=True, random_state=8432):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    skf = StratifiedKFold(n_splits=fold, shuffle=shuffle, random_state=random_state)

    X_train_folds = []
    y_train_folds = []

    X_test_folds = []
    y_test_folds = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for idx_train in train_index:
            if len(data.iloc[idx_train, 1]) > 1 and type(data.iloc[idx_train, 1]) == list:
                for element in data.iloc[idx_train, 1]:
                    X_train.append(element)
                    y_train.append(data.iloc[idx_train, 2])
            else:
                X_train.append(data.iloc[idx_train, 1])
                y_train.append(data.iloc[idx_train, 2])

        X_train_folds.append(X_train)
        y_train_folds.append(y_train)

        for idx_test in test_index:
            if len(data.iloc[idx_test, 1]) > 1:
                for element in data.iloc[idx_test, 1]:
                    X_test.append(element)
                    y_test.append(data.iloc[idx_test, 2])
            else:
                X_test.append(data.iloc[idx_test, 1])
                y_test.append(data.iloc[idx_test, 2])

        X_test_folds.append(X_test)
        y_test_folds.append(y_test)

    return X_train_folds, y_train_folds, X_test_folds, y_test_folds


if __name__ == '__main__':
    pat = make_list_of_patients()

    folds = cross_validation_splits(data=pat)

    print('ciao')
