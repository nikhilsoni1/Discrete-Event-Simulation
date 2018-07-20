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


def divide(obj):
    try:
        return obj/float(60)
    except (AttributeError, TypeError):
        return obj


def transform(df1, df2=None):
    df = df1
    df1 = df1.reset_index(drop=True)
    group = df1.groupby(by='Product')

    store_count = {'HAWK': list(), 'I2': list()}
    store_time = {'HAWK': list(), 'I2': list()}

    for i, j in group:
        store_time[i] = j['Time']
        group1 = j.groupby(by='Run')
        for k, l in group1:
            # print(l)
            store_count[i].append(len(l))
    for i in store_count:
        store_count[i] = ci(store_count[i])

    for i in store_time:
        store_time[i] = ci(store_time[i])

    df2 = df2.drop(columns=['Run', 'Count'])
    df2 = df2.apply(lambda x: divide(x))
    avg = {i: store_count[i]['Average'] for i in store_count}
    df2.insert(loc=1, column='Average Products', value=df2['Product'].map(avg))

    df_main = df2
    df_count_ci = pd.DataFrame(store_count).reindex(['High', 'Average', 'Low'])
    df_time_ci = (pd.DataFrame(store_time)/float(60)).reindex(['High', 'Average', 'Low'])
    return df_main, df_count_ci, df_time_ci


def ci(data, confidence=0.95):
    data = 1.0 * np.array(data)
    a = data[abs(data - np.mean(data)) < 1 * np.std(data)]
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1 + confidence) / 2., n - 1)
    return {'High': m+h, 'Average': m, 'Low': m - h}


def cell_1(id, product):
    global lay_down_op, stand_up_op
    global temp, temp1, temp2, temp3, temp4
    global flag, flag1, flag2
    global DSTAT_IA_IRON

    t1 = env.now

    #   Aluminium Bore----

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

    #   Aluminium Bore----

    #   Leak Test----

    if product.name.lower() == 'hawk':
        leak_req = yield leak_test.get(lambda machine: machine.op.lower() == 'hawk')
    elif product.name.lower() == 'i2':
        leak_req = yield leak_test.get(lambda machine: machine.op.lower() == 'i2')
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(5, 10))
    operator.release(op_req)
    yield env.timeout(r.uniform(LEAK[0], LEAK[1]))
    leak_test.put(leak_req)

    #   Leak Test----

    t3 = env.now

    #   Sleeve----

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

    #   Sleeve----

    t4 = env.now

    #   Build----

    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    if product.name.lower() == 'hawk':
        yield env.timeout(r.uniform(HAWK_BUILD[0], HAWK_BUILD[1]))
    elif product.name.lower() == 'i2':
        yield env.timeout(r.uniform(I2_BUILD[0], I2_BUILD[1]))
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)

    #   Build----

    t5 = env.now

    #   Iron Bore----

    sup1 = yield iron_bore.get(lambda machine: machine.op == 'SUP')

    if len([i for i in iron_bore.items if i.op == 'SUP']) == 0:
        temp.succeed()
    else:
        yield temp
    temp = env.event()

    if not flag:
        flag = True
        yield temp1
    elif flag:
        op_req = operator.request()
        yield op_req
        yield env.timeout(CHANGEOVER)
        operator.release(op_req)
        temp1.succeed()
        temp1 = env.event()
        flag = False

    yield env.timeout(SUP)

    ldn1 = yield iron_bore.get(lambda machine: machine.op == 'LDN')

    if not flag1:
        flag1 = True
        yield temp2
    elif flag1:
        op_req = operator.request()
        yield op_req
        yield env.timeout(CHANGEOVER)
        temp2.succeed()
        temp2 = env.event()
        flag1 = False
        operator.release(op_req)

    yield env.timeout(LDN)

    if not flag2:
        flag2 = True
        yield temp3
        yield iron_bore.put(ldn1)
        yield iron_bore.put(sup1)

    elif flag2:
        op_req = operator.request()
        yield op_req
        yield env.timeout(CHANGEOVER)
        temp3.succeed()
        temp3 = env.event()
        flag2 = False
        yield iron_bore.put(ldn1)
        yield iron_bore.put(sup1)
        operator.release(op_req)

    #   Iron Bore----

    t6 = env.now

    #   Hone----

    if product.name.lower() == 'hawk':
        hone_req = yield hone.get(lambda machine: machine.op.lower() == 'hawk')
    else:
        hone_req = yield hone.get(lambda machine: machine.op.lower() == 'i2')
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)
    yield env.timeout(r.uniform(HONE[0], HONE[1]))
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    yield env.timeout(r.uniform(10, 16))
    hone.put(hone_req)
    operator.release(op_req)

    #   Hone----

    t7 = env.now

    #   Build----

    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(BREAK[0], BREAK[1]))
    operator.release(op_req)

    #   Build----

    t8 = env.now

    #   Wash----

    wash_req = wash.request()
    yield wash_req
    if wash.count == wash_cap:
        temp4.succeed()
    yield temp4
    temp4 = env.event()
    op_req = operator.request()
    yield op_req
    yield env.timeout(8)
    operator.release(op_req)
    yield env.timeout(450)

    #   Wash----

    t9 = env.now

    #   Insepct----

    op_req = operator.request()
    yield op_req
    yield env.timeout(INSP)
    operator.release(op_req)

    wash.release(wash_req)
    if wash.count == 0:
        temp.succeed()
    yield temp
    temp = env.event()

    #   Inspect----

    t10 = env.now

    product.data_inp({'Run': simulation_no + 1,
                      'Al Bore': t2-t1,
                      'Leak Test': t3-t2,
                      'Sleeve': t4-t3,
                      'Build': t5-t4,
                      'Iron Bore': t6-t5,
                      'Hone': t7-t6,
                      'Break': t8-t7,
                      'Wash': t9-t8,
                      'Inspect': t10-t9,
                      'Time': t10-t1})


def cell_2(id, product):
    global lay_down_op, stand_up_op
    global temp, temp1, temp2, temp3, temp4
    global flag, flag1, flag2

    t1 = env.now

    #   Aluminium Bore----

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

    #   Aluminium Bore----

    #   Leak Test----

    if product.name.lower() == 'hawk':
        leak_req = yield leak_test.get(lambda machine: machine.op.lower() == 'hawk')
    elif product.name.lower() == 'i2':
        leak_req = yield leak_test.get(lambda machine: machine.op.lower() == 'i2')
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(5, 10))
    operator.release(op_req)
    yield env.timeout(r.uniform(LEAK[0], LEAK[1]))
    leak_test.put(leak_req)

    #   Leak Test----

    t3 = env.now

    #   Sleeve----

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

    #   Sleeve----

    t4 = env.now

    #   Build----

    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    if product.name.lower() == 'hawk':
        yield env.timeout(r.uniform(HAWK_BUILD[0], HAWK_BUILD[1]))
    elif product.name.lower() == 'i2':
        yield env.timeout(r.uniform(I2_BUILD[0], I2_BUILD[1]))
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)

    #   Build----

    t5 = env.now

    #   Iron Bore----

    sup1 = yield iron_bore.get(lambda machine: machine.op == 'SUP')
    sup2 = yield iron_bore.get(lambda machine: machine.op == 'SUP')

    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)
    yield env.timeout(120*correction)

    ldn1 = yield iron_bore.get(lambda machine: machine.op == 'LDN')
    ldn2 = yield iron_bore.get(lambda machine: machine.op == 'LDN')

    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)
    yield env.timeout(900*correction)

    yield iron_bore.put(ldn1)
    yield iron_bore.put(ldn2)
    yield iron_bore.put(sup1)
    yield iron_bore.put(sup2)

    #   Iron Bore----

    t6 = env.now

    #   Hone----

    if product.name.lower() == 'hawk':
        hone_req = yield hone.get(lambda machine: machine.op.lower() == 'hawk')
    else:
        hone_req = yield hone.get(lambda machine: machine.op.lower() == 'i2')
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    operator.release(op_req)
    yield env.timeout(r.uniform(HONE[0], HONE[1]))
    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(8, 10))
    yield env.timeout(r.uniform(10, 16))
    hone.put(hone_req)
    operator.release(op_req)

    #   Hone----

    t7 = env.now

    #   Build----

    op_req = operator.request()
    yield op_req
    yield env.timeout(r.uniform(BREAK[0], BREAK[1]))
    operator.release(op_req)

    #   Build----

    t8 = env.now

    #   Wash----

    wash_req = wash.request()
    yield wash_req
    if wash.count == wash_cap:
        temp4.succeed()
    yield temp4
    temp4 = env.event()
    op_req = operator.request()
    yield op_req
    yield env.timeout(8)
    operator.release(op_req)
    yield env.timeout(450)

    #   Wash----

    t9 = env.now

    #   Insepct----

    op_req = operator.request()
    yield op_req
    yield env.timeout(INSP)
    operator.release(op_req)

    wash.release(wash_req)
    if wash.count == 0:
        temp.succeed()
    yield temp
    temp = env.event()

    #   Inspect----

    t10 = env.now

    product.data_inp({'Run': simulation_no + 1,
                      'Al Bore': t2-t1,
                      'Leak Test': t3-t2,
                      'Sleeve': t4-t3,
                      'Build': t5-t4,
                      'Iron Bore': t6-t5,
                      'Hone': t7-t6,
                      'Break': t8-t7,
                      'Wash': t9-t8,
                      'Inspect': t10-t9,
                      'Time': t10-t1})


def process_wrapper(config=cell_1, worst=False):
    i = 0
    while True:
        if not worst:
            send = r.choice(prod_list)
        elif worst:
            send = prod_list[i % 2]
        env.process(config(id=i+1, product=send))
        i = i + 1
        yield env.timeout(r.uniform(2400, 2600))


pd.set_option('display.width', 1000)
pd.set_option("display.max_columns", 20)

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
SIM = 10
HONE = [220*correction, 300*correction]
BREAK = [19, 30]
WASH = [450, 490]
INSP = 40
LEAK = [35, 45]

UNTIL = shift(inp=1, opt='s')


r = Random()
# r.seed(34)
machine_obj = namedtuple('Machine', 'op')
prod_list = [Product(name='HAWK'), Product(name='I2')]

#   Simulation 1----

# for simulation_no in range(SIM):
#     env = simpy.Environment()
#     machine = simpy.FilterStore(env, capacity=3)
#     machine.items = [machine_obj(op='HQUALIFY'), machine_obj(op='HBORE'), machine_obj(op='I2BORE')]
#     leak_test = simpy.FilterStore(env, capacity=2)
#     leak_test.items = [machine_obj(op='HAWK'), machine_obj(op='I2')]
#     operator = simpy.PriorityResource(env, capacity=1)
#     sleeve = simpy.FilterStore(env, capacity=2)
#     sleeve.items = [machine_obj(op='HAWK'), machine_obj(op='I2')]
#     iron_bore = simpy.FilterStore(env, capacity=4)
#     iron_bore.items = [machine_obj(op='SUP'), machine_obj(op='LDN'), machine_obj(op='SUP'),
#                        machine_obj(op='LDN')]
#     hone = simpy.FilterStore(env, capacity=2)
#     hone.items = [machine_obj(op='HAWK'), machine_obj(op='I2')]
#     wash_cap = 2
#     wash = simpy.PriorityResource(env, capacity=wash_cap)
#
#     temp = env.event()
#     temp1 = env.event()
#     temp2 = env.event()
#     temp3 = env.event()
#     temp4 = env.event()
#     flag = False
#     flag1 = False
#     flag2 = False
#
#     env.process(process_wrapper(config=cell_1))
#     env.run(until=UNTIL)
#
#     del env, machine, leak_test, operator, sleeve, iron_bore, hone, wash, temp, temp1, temp2, temp3, temp4, flag,
#     flag1, flag2
#
# stat = Statistics(lst=prod_list)
# a, b, c = transform(stat.df, stat.mean())
# print('----------\nCONFIGURATION 1')
# print('%d Simulations\n' % (simulation_no + 1))
# print(a.round(1))
# print('\n Products Manufactured per Shift (95% Confidence)')
# print(b.round(2))
# print('\n Average time per product per shift (95% Confidence)')
# print(c.round(2))
# print('----------\n')

#   Simulation 1----

# Product.clear()
# del prod_list
# del stat
# prod_list = [Product(name='HAWK'), Product(name='I2')]

#   Simulation 2----

for simulation_no in range(SIM):
    env = simpy.Environment()
    machine = simpy.FilterStore(env, capacity=3)
    machine.items = [machine_obj(op='HQUALIFY'), machine_obj(op='HBORE'), machine_obj(op='I2BORE')]
    leak_test = simpy.FilterStore(env, capacity=2)
    leak_test.items = [machine_obj(op='HAWK'), machine_obj(op='I2')]
    operator = simpy.PriorityResource(env, capacity=1)
    sleeve = simpy.FilterStore(env, capacity=2)
    sleeve.items = [machine_obj(op='HAWK'), machine_obj(op='I2')]
    iron_bore = simpy.FilterStore(env, capacity=4)
    iron_bore.items = [machine_obj(op='SUP'), machine_obj(op='LDN'), machine_obj(op='SUP'),
                       machine_obj(op='LDN')]
    hone = simpy.FilterStore(env, capacity=2)
    hone.items = [machine_obj(op='HAWK'), machine_obj(op='I2')]
    wash_cap = 1
    wash = simpy.PriorityResource(env, capacity=wash_cap)

    temp = env.event()
    temp1 = env.event()
    temp2 = env.event()
    temp3 = env.event()
    temp4 = env.event()
    flag = False
    flag1 = False
    flag2 = False

    env.process(process_wrapper(config=cell_2))
    env.run(until=UNTIL)

    del env, machine, operator

stat = Statistics(lst=prod_list)
a, b, c = transform(stat.df, stat.mean())
print('----------\nCONFIGURATION 1')
print('%d Simulations\n' % (simulation_no + 1))
print(a.round(1))
print('\n Products Manufactured per Shift (95% Confidence)')
print(b.round(2))
print('\n Average time per product per shift (95% Confidence)')
print(c.round(2))
print('----------\n')

#   Simulation 2----

Product.clear()
del prod_list
del stat
prod_list = [Product(name='HAWK'), Product(name='I2')]