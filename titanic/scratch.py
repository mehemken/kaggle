#!/usr/bin/env python

import logging
logger = logging.getLogger()
logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(message)s'
    )
import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')

    df.head(1)

    g = df.groupby(['Embarked', 'Survived']).agg({'Survived':'sum'})
    pcts = g.groupby(level=0).apply(lambda x:
            x / float(x.sum()))
    print(pcts)
