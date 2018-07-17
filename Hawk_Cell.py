import pandas as pd
from scipy import stats
import os
import simpy
from random import Random
import numpy as np
from collections import namedtuple
import tempfile
import itertools as IT
from matplotlib import pyplot as plt


def ci(data, confidence=0.95):
    data = 1.0 * np.array(data)
    a = data[abs(data - np.mean(data)) < 1 * np.std(data)]
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def unique(pth, sep=''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s=sep, n=next(count))

    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        pth = os.path.normpath(pth)
        dirname, basename = os.path.split(pth)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename


class Hawk:
    class Container:
        def __init__(self, **kwargs):
            for i in kwargs:
                setattr(self, i, kwargs[i])

        def set(self, **kwargs):
            for i in kwargs:
                setattr(self, i, kwargs[i])

    def __init__(self, dir, dir1):
        self.dir = dir
        self.dir1 = dir1
        self.times = self.wrapper()

    def rm_outliers(self, df):
        df.insert(loc=len(df.columns), column='ZS', value=stats.zscore(a=df['MachiningTime']))
        df = df[df['ZS'].abs() < 3]
        df = df.drop(columns=['ZS'])
        return df

    def dtype_seconds(self, x):
        try:
            return x.dt.seconds
        except AttributeError:
            return x

    def machining_time(self, df):
        df_grouped = df.groupby(by='MainProgram')
        store = {}
        for i, j in df_grouped:
            j = j.apply(pd.to_timedelta, errors='ignore', unit='s')
            j = j.apply(lambda x: self.dtype_seconds(x))
            j = self.rm_outliers(j)
            mu = j['MachiningTime'].mean()
            sigma = j['MachiningTime'].std()
            store[str(i)] = Hawk.Container(mu=mu, sigma=sigma)
        return store

    def directory_crawler(self, path):
        st = list()
        for i in os.listdir(dir):
            if '.csv' in i.lower() and 'machi' in i.lower():
                j = '%s\%s' % (dir, i)
                st.append(pd.read_csv(j))
        df = pd.concat(st)
        return df

    def op_times(self, path):
        df = pd.read_csv(r"C:\Users\nikhiso\PycharmProjects\Machining\Input\Hawk_Cell\Time.csv")
        mean = df.mean().to_dict()
        sd = df.std().to_dict()
        store = {i: Hawk.Container(mu=mean[i], sigma=sd[i]) for i in mean}
        return store

    def wrapper(self):
        a = self.machining_time(self.directory_crawler(dir))
        b = self.op_times(self.dir1)
        c = a.copy()
        c.update(b)
        del a, b
        return c


class Product:
    MASTER_COUNT = 0
    MASTER_DATA = []
    MASTER_DF = None
    MASTER_INDEX = []

    @classmethod
    def MASTER_STATS(cls):
        if Product.MASTER_DATA:
            df = pd.DataFrame(Product.MASTER_DATA, columns=Product.MASTER_DATA[0].keys())
            df.insert(loc=0, column='Count', value=Product.MASTER_COUNT)
            df.index = np.arange(start=1, stop=Product.MASTER_COUNT + 1)
            Product.MASTER_DF = df
            return df

    @classmethod
    def clear(cls):
        Product.MASTER_COUNT = 0
        Product.MASTER_DATA = []
        Product.MASTER_DF = None
        Product.MASTER_INDEX = []

    def __init__(self, name):
        self.name = name
        self.count = 0
        self.data = []
        self.flag = 0

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
        Product.MASTER_DATA.append(row)
        Product.MASTER_COUNT = Product.MASTER_COUNT + 1


def shift(inp=float(1), opt='w'):
    if opt.lower() == 'w':
        return inp * 40 * 3600
    elif opt.lower() == 's':
        return inp * 8 * 3600


def request(resource, priority=None, delay=None):
    if priority:
        with resource.request(priority=priority) as req:
            yield req
            if delay:
                yield env.timeout(delay)
    else:
        with resource.request() as req:
            yield req
            if delay:
                yield env.timeout(delay)


def gauss(obj, correction=True):
    delay = -1
    while delay < 0 and correction:
        delay = r.gauss(obj.mu, obj.sigma)
        z = (delay - obj.mu) / obj.sigma
        z = abs(z)
        if z >= 0.5:
            delay = -1
    return delay


def cell(id, env, operator, times, product, machine):
    x = gauss(times['//PRO_MEM/O475'])

    t1 = env.now

    leak_req = leak_test.request()
    yield leak_req
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(10, 15))
    operator.release(op_req)
    yield env.timeout(gauss(times['LEAK_TEST']))
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(10, 15))
    leak_test.release(leak_req)
    operator.release(op_req)

    t2 = env.now

    sleeve_req = sleeve.request()
    yield sleeve_req
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(10, 15))
    operator.release(op_req)
    yield env.timeout(gauss(times['SLEEVE']))
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(10, 15))
    sleeve.release(sleeve_req)
    operator.release(op_req)

    t3 = env.now
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(5, 7))
    yield env.timeout(gauss(times['BUILD']))
    operator.release(op_req)

    t4 = env.now

    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(5, 7))
    yield env.timeout(gauss(times['MACHINE_SCREW']))
    operator.release(op_req)

    t5 = env.now

    req1 = yield machine.get(lambda machine: machine.op == 'stand_up')
    op_req = operator.request()
    yield op_req
    yield env.timeout(gauss(times['OP1_OP2_CHANGEOVER']))
    operator.release(op_req)
    yield env.timeout(x)
    op_req = operator.request()
    yield op_req
    yield env.timeout(gauss(times['OP1_OP2_CHANGEOVER']))
    req2 = yield machine.get(lambda machine: machine.op == 'lay_down')
    machine.put(req1)
    operator.release(op_req)
    yield env.timeout(x)
    op_req = operator.request()
    yield op_req
    yield env.timeout(gauss(times['OP1_OP2_CHANGEOVER']))
    machine.put(req2)
    operator.release(op_req)

    t6 = env.now

    hone_req = hone.request()
    yield hone_req
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(10, 12))
    operator.release(op_req)
    yield env.timeout(gauss(times['HONE']))
    op_req = operator.request()
    yield op_req
    yield env.timeout(gauss(times['GAUGE']))
    yield env.timeout(gauss(times['HONE_UNMOUNT']))
    hone.release(hone_req)
    operator.release(op_req)

    t7 = env.now

    op_req = operator.request()
    yield op_req
    yield env.timeout(gauss(times['BREAK']))
    operator.release(op_req)

    t8 = env.now

    wash_req = wash.request()
    yield wash_req
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)
    yield env.timeout(gauss(times['WASH']))
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(15, 18))
    wash.release(wash_req)
    operator.release(op_req)

    t9 = env.now

    row = {'ID': id,
           'Total': t9 - t1,
           't1': t1,
           't2': t2,
           't3': t3,
           't4': t4,
           't5': t5,
           't6': t6,
           't7': t7,
           't8': t8,
           't9': t9}
    # print(id, row)
    product.data_inp(row)


def wrapper(env):
    i = 0
    while True:
        env.process(cell(i + 1, env, operator, times, Product('Hawk'), a61))
        i = i + 1
        yield env.timeout(r.uniform(1300, 1500))
        # yield env.timeout(r.uniform(750, 1000))


pd.set_option('display.width', 1000)
pd.set_option("display.max_columns", 100)

a61_store = namedtuple('A61', 'op')
dir = r"C:\Users\nikhiso\PycharmProjects\Machining\Input\Hawk_Cell\A61\Result"
dir1 = r"C:\Users\nikhiso\PycharmProjects\Machining\Input\Hawk_Cell\Time.csv"
r = Random()
times = Hawk(dir, dir1).times

store = []

for i in range(1000):
    env = simpy.Environment()
    operator = simpy.PriorityResource(env, capacity=1)
    leak_test = simpy.PriorityResource(env, capacity=1)
    sleeve = simpy.PriorityResource(env, capacity=1)
    a61 = simpy.FilterStore(env, capacity=2)
    a61.items = [a61_store(op='stand_up'), a61_store(op='lay_down')]
    hone = simpy.PriorityResource(env, capacity=1)
    wash = simpy.PriorityResource(env, capacity=2)

    env.process(wrapper(env))
    env.run(until=shift(inp=0.2))
    df = Product.MASTER_STATS()
    j = 0
    try:
        j = df['Count'].mean()
        store.append(j)
    except TypeError:
        pass
    Product.clear()
    del env, operator, leak_test, sleeve, a61, hone, wash, df, j

mid, lower, upper = ci(store)
print('%.2f - %.2f - %.2f' % (lower, mid, upper))






