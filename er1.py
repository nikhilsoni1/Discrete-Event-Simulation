import pandas as pd
import simpy
from random import Random
from collections import namedtuple
import numpy as np
import tempfile
import itertools as IT
import os
from scipy import stats
import matplotlib.pyplot as plt
import math


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def unique(pth, sep=''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        pth = os.path.normpath(pth)
        dirname, basename = os.path.split(pth)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename


class Matrix:
    def __init__(self, path):
        self.path = path
        self.matrix = self.matrix()
        self.df = pd.read_excel(path, sheet_name=1)

    def products(self):
        store = list(set(self.df['Part'].tolist()))
        return [Product(i) for i in store]

    def product_machine_palette(self):
        file = pd.ExcelFile(self.path)
        df = pd.read_excel(file, sheet_name=1)
        df = df[['Order', 'Palette', 'MC']]
        df_dict = df.transpose().to_dict()
        store = [list(df_dict[i].values()) for i in df_dict]
        store = [''.join(str(i).strip('[]')) for i in store]
        store = list(set(store))
        store = [list(map(int, i.split(','))) for i in store]
        product_machine_palette = dict()
        for i in store:
            if str(i[0]) in product_machine_palette:
                product_machine_palette[str(i[0])].append({'mc': i[2], 'p': i[1]})
            else:
                product_machine_palette[str(i[0])] = [{'mc': i[2], 'p': i[1]}]
        return product_machine_palette

    def product_machine_palette_time(self):
        xlsx = pd.ExcelFile(self.path)
        df = pd.read_excel(xlsx, sheet_name=1)
        df['MP'] = df['MC'].astype(str) + "," + df['Palette'].astype(str)
        df = df[['Part', "Duration", 'MP']]
        df = self.rm_outliers(df=df)
        df_grouped = df.groupby(by='Part')
        product_machine_palette_time_mean = dict()
        product_machine_palette_time_std = dict()
        for i, j in df_grouped:
            a = j.groupby(by='MP')['Duration'].mean()
            b = j.groupby(by='MP')['Duration'].std()
            product_machine_palette_time_mean[str(i)] = a.to_dict()
            product_machine_palette_time_std[str(i)] = b.to_dict()
        return product_machine_palette_time_mean, product_machine_palette_time_std

    def matrix(self):
        a = self.product_machine_palette()
        b, c = self.product_machine_palette_time()
        for i in a:
            for j in a[i]:
                val = list(j.values())
                val = list(map(str, val))
                val = ','.join(val)
                j['t'] = round(b[i][val], 3)
                j['sd'] = round(c[i][val], 3)
        return a

    def get(self, key, option='p', palette=None, machine=None):
        key = str(key)
        if key in self.matrix:
            data = pd.DataFrame(self.matrix[key])
            df = data.pivot(index='mc', columns='p', values='t')
            df1 = data.pivot(index='mc', columns='p', values='sd')
            if option.lower() == 'p':
                return df.columns.values.tolist()
            elif option.lower() == 'm' and palette is not None:
                return df[df[palette].notnull()].index.values.tolist()
            elif option.lower() == 't' and palette is not None and machine is not None:
                return df.ix[machine, palette], df1.ix[machine, palette]

    def rm_outliers(self, df):
        df_store = []
        df_store_outliers_removed = []
        df_part = df.groupby(by='Part')
        for i, j in df_part:
            for p, q in j.groupby(by='MP'):
                df_store.append(q)
        for i in df_store:
            i.insert(loc=len(i.columns), column='ZS', value=stats.zscore(a=i['Duration']))
            i = i[i['ZS'].abs() < 3]
            i = i.drop(columns='ZS')
            df_store_outliers_removed.append(i)
        return pd.concat(df_store_outliers_removed)


class HeteroResource:
    def __init__(self, env, name, cap, **kwargs):
        self.kwargs = kwargs
        self.apparatus = namedtuple(name, 'num')
        self.env = env
        self.cap = cap
        self.store = simpy.FilterStore(env, capacity=cap)
        self.store.items = [self.apparatus(i) for i in range(1, cap + 1)]
        self.capacity()
        self.requests = dict()
        self.wait_request = dict()
        self.wait_times = dict()

    def data(self, effects):
        if effects[1] == 0:
            for i in effects[0]:
                if str(i) in self.requests:
                    self.requests[str(i)] = self.requests[str(i)] + 1
                else:
                    self.requests[str(i)] = 1
        elif effects[1] != 0:
            for i in effects[0]:
                if str(i) in self.wait_request:
                    self.wait_request[str(i)] = self.wait_request[str(i)] + 1
                    self.wait_times[str(i)].append(effects[1])
                else:
                    self.wait_request[str(i)] = 1
                    self.wait_times[str(i)] = [effects[1]]

    def capacity(self):
        if self.kwargs:
            store = list(self.kwargs.values())[0]
            items = [self.apparatus(i) for i in range(1, self.cap + 1)]
            increase_cap = sum(list(store.values()))-len(store)
            self.cap = self.cap + increase_cap
            self.store = simpy.FilterStore(self.env, capacity=self.cap)
            for i in store:
                for j in range(store[i]-1):
                    items.append(self.apparatus(int(i)))
            self.store.items = items

    def resource_load(self):
        df1 = pd.DataFrame({'Load': self.requests})
        df2 = pd.DataFrame({'Load': self.wait_request})
        df3 = df1.sub(other=df2)
        df4 = df2.div(other=df3).round(2)
        df4 = df4[df4['Load'].notna()]
        df4.set_index(keys=df4.index.astype('int'), inplace=True)
        df4.sort_index(inplace=True)
        df4['Mean'] = df4.index.astype('str').map({i: np.mean(self.wait_times[i]) for i in self.wait_times})
        df4['Sigma'] = df4.index.astype('str').map({i: np.std(self.wait_times[i]) for i in self.wait_times})
        df4['Min.'] = df4.index.astype('str').map({i: np.min(self.wait_times[i]) for i in self.wait_times})
        df4['Max.'] = df4.index.astype('str').map({i: np.max(self.wait_times[i]) for i in self.wait_times})
        return df4


class Resource:
    def __init__(self, env, cap):
        self.resource = simpy.Resource(env=env, capacity=cap)
        self.queue = []

    def data(self):
        self.queue.append(len(self.resource.queue))

    def mean(self):
        return math.ceil(float(sum(self.queue))/float(len(self.queue)))


class Product:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.data = []

    def stats(self):
        if self.data:
            df = pd.DataFrame(self.data, columns=self.data[0].keys())
            df.insert(loc=0, column='Count', value=self.count)
            df.name = self.name
            df.index = np.arange(start=1, stop=self.count + 1)
            return df

    def data_inp(self, row):
        self.data.append(row)
        self.count = self.count + 1


class Statistics:
    def __init__(self, lst):
        self.products = lst
        self.df = self.concatenate_product_times(lst)

    def concatenate_product_times(self, artefacts):
        store = []
        for i in artefacts:
            df = i.stats()
            if df is not None:
                df.insert(loc=0, column="Product", value=df.name)
                store.append(df)
        df = pd.concat(store)
        return df

    def mean(self):
        df = self.df.groupby(by='Product').mean()
        df = df.reset_index()
        df.index = np.arange(start=1, stop=len(df) + 1)
        print('Total Products Machined: %d' % df['Count'].sum())
        return df

    def norm_prob_plot(self):
        df = self.df.groupby(by='Product')
        for i, j in df:
            stats.probplot(j['Total Time'], plot=plt, dist='norm')
            plt.show()


def op(i, product, env, palette_object, machine_object, wss, operator):
    machine_store = machine_object.store
    palette_store = palette_object.store
    
    palette = mat.get(key=product.name, option='p')

    a = env.now
    palette_object.data(effects=(palette, 0))
    p = yield palette_store.get(lambda foo: foo.num in palette)
    b = env.now
    if b - a > 0:
        palette_object.data(effects=(palette, b-a))

    # print(i + 1, ").", product.name, 'got', p, 'at', b)
    
    machine = mat.get(key=product.name, option='m', palette=p[0])
    
    machine_object.data(effects=(machine, 0))
    m = yield machine_store.get(lambda foo: foo.num in machine)
    c = env.now
    if c - b > 0:
        machine_object.data(effects=(machine, c-b))

    # print(i + 1, ").", product.name, 'got', m, 'at', c)

    mu, sigma = mat.get(key=product.name, option='t', palette=p[0], machine=m[0])

    delay = abs(r.gauss(mu=mu, sigma=sigma))
    yield env.timeout(delay)
    d = env.now
    yield machine_store.put(m) and palette_store.put(p)
    with wss.resource.request() as req:
        wss.data()
        yield req
        yield env.timeout(r.expovariate(1/240))
    e = env.now
    with operator.resource.request() as req:
        operator.data()
        yield req
        yield env.timeout(r.expovariate(1/900))
    f = env.now

    product.data_inp({'PWT': b-a,
                     'MST': c-b,
                    'MT': d-c,
                     'WST': e-d,
                     'OST': f-e,
                     'Total Time': f-a})

    # print(i + 1, ").", product.name, 'released', p, 'and', m, 'at', e)


def wrapper(env):
    i = 0
    while True:
        name = products[r.randint(0, len(products) - 1)]
        env.process(op(i, name, env, palette, er1, wss, operator))
        i = i + 1
        yield env.timeout(r.uniform(600, 700))


def time(inp=1, opt='w'):
    if opt.lower() == 'w':
        return inp*120*3600
    elif opt.lower() == 's':
        return inp*8*3600


pd.set_option('display.width', 1000)
pd.set_option("display.max_columns", 10)

env = simpy.Environment()

path = 'Input/ER1/ER1_Match_Matrix.xlsx'

r = Random()
r.seed(9)

# extra_palette = {'14': 2, '3': 4, '6': 2, '11': 2}
er1 = HeteroResource(env=env, name='Machine', cap=4)
palette = HeteroResource(env=env, name='Palette', cap=15)
wss = Resource(env=env, cap=2)
operator = Resource(env=env, cap=3)
mat = Matrix(path)
products = mat.products()

env.process(wrapper(env))
env.run(until=time(inp=1))

stat = Statistics(products)
print(stat.mean())

print('WSS Average Queue Length: %d' % wss.mean())
print('Operator Average Queue Length: %d' % operator.mean())
print('')
print(palette.resource_load())
print(er1.resource_load())