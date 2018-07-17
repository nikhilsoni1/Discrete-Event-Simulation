import pyodbc
import pandas as pd
import tempfile
import itertools as IT
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def unique(path, sep=''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename


class Db:
    def __init__(self, server, database, username, password, driver, port):
        self.credential = 'DRIVER= %s;SERVER=%s;PORT=%s;DATABASE=%s;WSID=USST-6FVYQ22; ' \
              'APP=Microsoft Office 2016;UID=%s;PWD=%s' % (driver, server, port, database, username, password)
        self.connection = self.connect()
        self.cursor = self.connection.cursor()
        mkdir('output')

    def connect(self):
        return pyodbc.connect(self.credential)

    def close(self):
        self.cursor.close()
        del self.cursor
        self.connection.close

    def table_names(self, option, search=None):
        flag = 0
        if option == 1:
            return self.cursor.tables()
        elif option == 2:
            for row in self.cursor.tables():
                if search.lower() in row.table_name.lower():
                    flag = flag + 1
                    print(row.table_name)
            if flag > 0:
                print("\n%d results found which contain \"%s\"" % (flag, search))
            else:
                print("No results found")
        else:
            print('Bad Selection')


def data(from_date, to_date, opt=None):
    server = '255.255.255.255'
    database = 'DB'
    username = 'johndoe'
    password = '12345678'
    driver = '{SQL Server}'
    port = '1433'
    g = Db(server, database, username, password, driver, port)
    query1 = "SELECT * FROM MACH_HISTORY \
                   WHERE CREATE_DATE BETWEEN '%s' AND '%s'" % (from_date, to_date)
    df_history = pd.read_sql(query1, g.connection)
    df_history = df_history.dropna(axis=1, how='all')
    df_product = product_lookup(vector=df_history['MACH_ID'], connection=g.connection)
    df = df_history.merge(df_product, on='MACH_ID', how='left')
    del df_history
    del df_product
    g.close()
    df = df[['REC_ID',
             'MACH_ID',
             'RPT_PROD_DESC',
             'ASSY_NO',
             'CREATE_DATE',
             'OP1_DATE',
             'SLEEVE_DATE',
             'CRANKCOVER_DATE',
             'OP2_DATE',
             'OP3_DATE',
             'HONE_DATE',
             'DEBURR_DATE',
             'INSPECT_DATE',
             'CONSUMED_DATE']]
    df = df.dropna(axis=0, how='any', subset=['RPT_PROD_DESC'])
    if opt == 1:
        df.to_excel(unique("output/Data.xlsx"), sheet_name='Sheet 1', index=False)
        print(df.head(5))
    elif opt == 2:
        print(df.head(5))
    return df


def product_lookup(vector, connection):
    store = []
    vector = vector.tolist()
    chunks = [vector[x:x + 25000] for x in range(0, len(vector), 25000)]
    for i in chunks:
        query = "SELECT MACH_HISTORY.MACH_ID, RPT_FAMILY.RPT_PROD_DESC \
             FROM (MACH_HISTORY INNER JOIN HISTORY ON MACH_HISTORY.MACH_ID = HISTORY.RAW_PH_ID) \
            INNER JOIN (MODEL INNER JOIN RPT_FAMILY ON MODEL.RPT_FAMILY_NO = RPT_FAMILY.RPT_FAMILY_NO) \
            ON HISTORY.MODEL_ENG = MODEL.MODEL_ENG \
            WHERE MACH_HISTORY.MACH_ID in %s;" % repr(tuple(map(str, i)))
        store.append(pd.read_sql(sql=query, con=connection))
    return pd.concat(store)


def plot(df, column):
    grouped = df.groupby(column)
    k = 1
    l = 20
    b = l/1.29
    plt.figure(figsize=(l, b), dpi=300)
    plt.suptitle(t='Machining | Total Lead Time\n%s to %s' % (from_date.strftime("%b-%d-%y"),
                                                             to_date.strftime("%b-%d-%y")), fontsize='16')
    # logo = plt.imread('Input/brp_logo.png')
    # plt.figimage(X=logo)
    for i, j in grouped:
        mean = round(j['Total'].mean(), 2)
        median = round(j['Total'].median(), 2)
        plt.subplot(2, 3, k, facecolor='#f7f7f7')
        plt.hist(x=j['Total'], bins=500)
        ymin, ymax = plt.ylim()
        plt.axvline(x=mean, color='k', linestyle='dashed', label='Average')
        plt.axvline(x=median, color='g', linestyle='dashed', label='Median')
        plt.text(mean*1.1, ymax*0.85, str(mean), bbox=dict(fc="k", boxstyle='round,pad=0.15'), color='w', )
        plt.text(median*1.1, ymax * 0.75, str(median), bbox=dict(fc="#57cc65", boxstyle='round,pad=0.15'))
        plt.xlabel('Lead Time (Hours)')
        plt.ylabel('Number of Products')
        plt.legend(loc=1)
        plt.title(i, fontsize='x-large')
        k = k + 1
    stamp = '%s_%s' % (from_date.strftime('%m%d%y'), to_date.strftime('%m%d%y'))
    mkdir('output/plots')
    plt.savefig(fname=unique('output/plots/%s_lead_time.png' % stamp))


def rm_outliers(df, group_by, target_column):
    grouped = df.groupby(by=group_by)
    combine = []
    for i, j in grouped:
        std = j[target_column].std()
        std = std*3
        j = j[j[target_column] < std]
        combine.append(j)
    df = pd.concat(combine, ignore_index=True)
    return df


def lead_time(df, value=None, option=None, quantity='q'):
    factor = pow(10, 11) * 36
    counter = len(df.columns.values)-1
    while counter > 4:
        store = (df.ix[:, counter]-df.ix[:, counter-1]).astype('timedelta64')/factor
        df["%d-%d" % (counter + 1, counter)] = store
        counter = counter - 1
        # df.insert(loc=counter, column=str(counter-1), value=store)
        # counter = counter - 1
    df['Total'] = (df.ix[:, 12]-df.ix[:, 4]).astype('timedelta64')/factor
    df = df.iloc[:, np.r_[2, 15:len(df.columns.values)]]
    num = df._get_numeric_data()
    num[num < 0] = np.nan
    df = rm_outliers(df, 'RPT_PROD_DESC', 'Total')
    plot(df, "RPT_PROD_DESC")
    if quantity.lower() in 'm':
        df = df.groupby('RPT_PROD_DESC').mean()
    elif quantity.lower in 'q' or quantity is None:
        df = df.groupby('RPT_PROD_DESC').quantile(value if value is not None else 0.96)

    df = df[['13-12', '12-11', '11-10', '10-9', '9-8', '8-7', '7-6', '6-5', 'Total']]
    if option == 1:
        df.to_excel(unique('output/quant.xlsx'))
        print(df)
    return df


from_date = dt(year=2018, month=1, day=1, hour=0, minute=0, second=0)
to_date = dt(year=2018, month=3, day=31, hour=23, minute=59, second=59)

past = dt.now()
lead_time(df=data(from_date=from_date, to_date=to_date), quantity='m')
present = dt.now()
print(present-past)
