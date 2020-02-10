import pandas as pd
import json

class ReadData(object):
    '''

    Read list of objects from a JSON lines file and converts the list to pandas dataframe.

    Arguments:
        train data filepath: string 
        test data filepath: string
    '''
    def __init__(self, input_file):

        self.input_file = input_file

    def read_data(self):

        data = []

        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.rstrip('\n|\r')))
        print('Loaded {} records from {}'.format(len(data), self.input_file))

        df = pd.DataFrame(data)

        return df
        


