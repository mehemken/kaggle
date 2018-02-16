import pandas as pd
import os
import logging
logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(message)s'
)


class TitanicETL:
    '''This is the ETL class I'll use for the rest of this project.
    It will be used to load the data and make necessary transformations.
    '''
    def __init__(self, location=''):
        if not location:
            logger.fatal('Please specify a file.')
            return None
        try:
            self.location = os.path.abspath(location)
        except:
            logger.exception('Could not load file.')
        return
    
    def _load_from_file(self):
        '''Loads data from a csv file and sets
        the index to PassengerId.'''
        df = pd.read_csv(self.location)
        return df.set_index('PassengerId')
    
    def _add_has_cabin(self, df):
        '''Adds a boolean column corresponding
        to a passenger having/not a cabin.'''
        df['HasCabin'] = df['Cabin'].notna()
        return df
    
    def _get_cabin_letters(self, df):
        '''Takes the first letter of a passenger's cabin
        and creates a one-hot encoded matrix out of it.
        This gets left joined to the data.'''
        def get_letter(x):
            try:
                if x:
                    return 'C' + x[:1]
            except TypeError:
                logger.debug('float not subscriptable')
            except:
                logger.exception('Failed.')
            return 'C0'

        s = df['Cabin'].apply(get_letter)
        t = pd.get_dummies(s)
        return df.join(t)
    
    def _get_embarked_letters(self, df):
        '''Creates a one-hot encoded matriks out of the
        letter in the Embarked column. This gets left
        joined to the data.'''
        s = pd.get_dummies(df['Embarked'])
        return df.join(s)
    
    def _get_sex_values(self, df):
        '''Creates a 1/0 value out of the passenger's sex.'''
        t = df['Sex'].astype('category')
        df['NSex'] = t.values.codes
        return df
    
    def _drop_columns(self, df):
        '''We don't need these anymore.'''
        df = df.drop('Age', axis=1)
        df = df.drop('Unnamed: 0', axis=1)
        df = df.drop('Sex', axis=1)
        df = df.drop('Embarked', axis=1)
        df = df.drop('Cabin', axis=1)
        df = df.drop('Name', axis=1)
        df = df.drop('Ticket', axis=1)
        return df
    
    def get(self):
        '''Takes all the transformations and puts
        them in a neat pipeline.'''
        return (
            self._load_from_file()
            .pipe(self._add_has_cabin)
            .pipe(self._get_cabin_letters)
            .pipe(self._get_embarked_letters)
            .pipe(self._get_sex_values)
            .pipe(self._drop_columns)
        )