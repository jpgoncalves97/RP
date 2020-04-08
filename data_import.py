import arff
import pandas as pd
from os import listdir
from os.path import join


def read_activities():
    with open(join('wisdm-dataset', 'activity_key.txt'), 'r') as file:
        data = file.read()
        data = data.split('\n')
        activities = {}
        for row in data:
            row = row.split(',')
            activities[row[1]] = (row[0], row[2])
    return pd.DataFrame.from_dict(activities, orient='index', columns=['desc', 'group'])


def arff_to_df(path):
    dt = arff.load(open(path))
    return pd.DataFrame(dt['data'], columns=[row[0] for row in dt['attributes']]).drop(['class'], axis=1)


def read_files(path, device, sensor, act_df):
    print('Reading', device, sensor)
    cur_path = join(path, device, sensor)
    df = pd.concat((arff_to_df(join(cur_path, f)) for f in listdir(cur_path) if f.endswith('.arff')))
    return df


def fix_arff_files(self):
    for d in ['phone', 'watch']:
        for s in ['accel', 'gyro']:
            cur_path = join(self.path, d, s)
            files = [f for f in listdir(cur_path) if f.endswith('.arff')]
            for f in files:
                with open(join(cur_path, f), 'r+') as file:
                    data = file.read()
                    data.replace('{ ', '{')
                    data.replace(' }', '}')
                    file.seek(0)
                    file.truncate()
                    file.write(data)
