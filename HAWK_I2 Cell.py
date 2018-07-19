import simpy
import pandas as pd
import numpy as np
from collections import namedtuple
from random import Random
from scipy import stats
from matplotlib import pyplot as plt


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
        self.id = 0
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
        # print('Total Products Machined: %d' % df['Count'].sum())
        return df

    def ci(self, confidence=0.95, by=None, value=None):
        store = dict()
        if by:
            group = self.df.groupby(by=by)
            for i, j in group:
                data = 1.0 * np.array(j[value].tolist())
                a = data[abs(data - np.mean(data)) < 1 * np.std(data)]
                n = len(a)
                m, se = np.mean(a), stats.sem(a)
                h = se * stats.t._ppf((1 + confidence) / 2., n - 1)
                store[i] = {'Upper': m+h, 'Mid': m, 'Lower': m-h}
        return store


    def norm_prob_plot(self):
        df = self.df.groupby(by='Product')
        for i, j in df:
            stats.probplot(j['Total Time'], plot=plt, dist='norm')
            plt.show()


def shift(inp=float(1), opt='w'):
    if opt.lower() == 'w':
        return inp * 40 * 3600
    elif opt.lower() == 's':
        return inp * 8 * 3600


def transform(lst):
    store = dict()
    total_count = 0
    total_time = 0
    for i in lst:
        store[i['Product']] = {'Count': i['Count'],
                               'Time': i['Time']}
        total_count = total_count + i['Count']
        total_time = total_time + i['Time']
    mean_time = total_time/float(len(lst))
    store['Total'] = {'Count': total_count, 'Time': mean_time}
    return store


def collector(obj):
    return True


def cell(id, product):
    t5 = -1000
    t1 = env.now

    if product.name == 'HAWK':
        req_I2BORE = yield machine.get(lambda machine: machine.op == 'I2BORE')
        req_HQUALIFY = yield machine.get(lambda machine: machine.op == 'HQUALIFY')
        op_req = operator.request()
        yield op_req
        yield env.timeout(CHANGEOVER)
        operator.release(op_req)
        yield env.timeout(HQUALIFY)
        req_HBORE = yield machine.get(lambda machine: machine.op == 'HBORE')
        op_req = operator.request()
        yield op_req
        yield env.timeout(CHANGEOVER)
        yield machine.put(req_HQUALIFY)
        operator.release(op_req)
        yield env.timeout(HBORE)
        op_req = operator.request()
        yield op_req
        yield env.timeout(CHANGEOVER)
        yield machine.put(req_HBORE)
        yield machine.put(req_I2BORE)
        operator.release(op_req)

        t2 = env.now

    elif product.name == 'I2':
        req_HQUALIFY = yield machine.get(lambda machine: machine.op == 'HQUALIFY')
        req_HBORE = yield machine.get(lambda machine: machine.op == 'HBORE')
        req_I2BORE = yield machine.get(lambda machine: machine.op == 'I2BORE')
        op_req = operator.request()
        yield op_req
        yield env.timeout(CHANGEOVER)
        operator.release(op_req)
        yield env.timeout(I2BORE)
        op_req = operator.request()
        yield op_req
        yield env.timeout(CHANGEOVER)
        yield machine.put(req_HQUALIFY)
        yield machine.put(req_HBORE)
        yield machine.put(req_I2BORE)
        operator.release(op_req)

        t2 = env.now

    sleeve_req = yield sleeve.get(lambda machine: machine.op == product.name)
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)
    yield env.timeout(r.uniform(SLEEVE[0], SLEEVE[1]))
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)
    yield sleeve.put(sleeve_req)

    t3 = env.now

    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    if product.name.lower() == 'hawk':
        yield env.timeout(r.uniform(HAWK_BUILD[0], HAWK_BUILD[1]))
    elif product.name.lower() == 'i2':
        yield env.timeout(r.uniform(I2_BUILD[0], I2_BUILD[1]))
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)

    t4 = env.now

    print(iron_bore.items)

    sup1 = yield iron_bore.get(lambda machine: machine.op == 'SUP')
    sup2 = yield iron_bore.get(lambda machine: machine.op == 'SUP')
    op_req = operator.request()
    yield op_req
    yield env.timeout(CHANGEOVER)
    operator.release(op_req)
    yield env.timeout(SUP)
    ldn1 = yield iron_bore.get(lambda machine: machine.op == 'LDN')
    ldn2 = yield iron_bore.get(lambda machine: machine.op == 'LDN')
    op_req = operator.request()
    yield op_req
    yield env.timeout(CHANGEOVER)
    yield iron_bore.put(sup1)
    yield iron_bore.put(sup2)

    operator.release(op_req)
    yield env.timeout(LDN)
    op_req = operator.request()
    yield op_req
    yield env.timeout(CHANGEOVER)
    yield iron_bore.put(ldn1)
    yield iron_bore.put(ldn2)
    operator.release(op_req)

    t5 = env.now

    hone_req = hone.request()
    yield hone_req
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)
    yield env.timeout(r.uniform(HONE[0], HONE[1]))
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    yield env.timeout(r.uniform(10, 16))
    hone.release(hone_req)
    operator.release(op_req)

    t6 = env.now

    product.data_inp({'Al Bore': t2-t1,
                      'Sleeve': t3-t2,
                      'Build': t4-t3,
                      'Iron Bore': t5-t4,
                      'Hone': t6-t5,
                      'Time': t6-t1})


def wrapper(worst=False):
    i = 0
    while True:
        if not worst:
            send = r.choice(prod_list)
        elif worst:
            send = prod_list[i % 2]
        # print('%d. %s' % (i+1, send.name))
        env.process(cell(id=i+1, product=send))
        i = i + 1
        yield env.timeout(r.uniform(1300, 2000))


correction = 1
HQUALIFY = float(14*60*correction)
HBORE = float(12*60*correction)
I2BORE = float(26*60*correction)
SUP = float(30*60*correction)
LDN = float(30*60*correction)
CHANGEOVER = float(120)
SLEEVE = [48, 80]
HAWK_BUILD = [86, 100]
I2_BUILD = [100, 150]
SIM = 100
HONE = [220*correction, 300*correction]

r = Random()
r.seed(34)
machine_obj = namedtuple('Machine', 'op')
prod_list = [Product(name='HAWK'), Product(name='I2')]

store = dict()
temp = list()
for i in range(SIM):
    # prod_list = [Product(name='HAWK'), Product(name='I2')]
    env = simpy.Environment()
    machine = simpy.FilterStore(env, capacity=6)
    machine.items = [machine_obj(op='HQUALIFY'), machine_obj(op='HBORE'), machine_obj(op='I2BORE')]
    operator = simpy.PriorityResource(env, capacity=1)
    sleeve = simpy.FilterStore(env, capacity=2)
    sleeve.items = [machine_obj(op='HAWK'), machine_obj(op='I2')]
    iron_bore = simpy.FilterStore(env, capacity=4)
    iron_bore.items = [machine_obj(op='SUP'), machine_obj(op='LDN'), machine_obj(op='SUP'),
                       machine_obj(op='LDN')]
    hone = simpy.PriorityResource(env, capacity=1)

    env.process(wrapper())
    env.run(until=shift(inp=1, opt='s'))

    del env, machine, operator#, prod_list, stat, lst, df


stat = Statistics(lst=prod_list)
print('Simulations: %d' % (i+1))
mean = stat.mean()
mean.loc[:, 'Count'] = mean.loc[:, 'Count']/float(SIM)
mean.loc[:, 'Time'] = mean.loc[:, 'Time']/float(60)
print(mean.round(2))
time_ci = pd.DataFrame(stat.ci(by='Product', value='Time'))
print((time_ci/60).round(2))
