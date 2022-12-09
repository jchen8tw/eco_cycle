import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime as dt
from utils import *
from typing import *
from enum import Enum


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

#################################   讀取資料   ####################################
data = pd.read_csv('./data/cy_index1124.csv')
data["date"] = pd.to_datetime(data["date"])


#################################   產生各指標時間序列  ###########################
class DataType(Enum):
    OECD = 'oecd'
    CRB = 'crb'
    US30R = 'us30r'


print(globals()['oecd'])


def genSignal(
        data_type: DataType,
        data: pd.Series,
        date: pd.Series,
        interval_1: int,
        interval_2: int) -> pd.DataFrame:
    # arr = np.array(data[data_type.value])
    arr = np.array(data)
    # namespaces: oecd crb us30r
    # e.g: globals()['oecd'].findLocal = oecd.findLocal
    mx, mn = globals()[data_type.value].findLocal(
        arr, interval_1=interval_1, interval_2=interval_2)

    print(mx)
    print(mn)
    # business cycle

    mx_index = mx[0:(len(mx))]
    mn_index = mn[0:(len(mn))]

    two_array1 = np.vstack((mx_index, np.ones(len(mx)))).T
    two_array2 = np.vstack((mn_index, np.zeros(len(mn)))).T

    df1 = pd.DataFrame(two_array1, columns=['index', 'cycle'])
    df2 = pd.DataFrame(two_array2, columns=['index', 'cycle'])

    me = pd.merge(df1, df2, how="outer", sort=True)

    tcy = []
    cy = me['cycle']
    id = me['index']

    if (cy[0] == 0):
        tcy.extend([0] * int(id[0]))

    elif (cy[0] == 1):
        tcy.extend([1] * int(id[0]))

    for i in range(1, (len(cy))):

        if (cy[(i - 1)] == 0 and cy[i] == 1):
            tcy.extend([1] * (int(id[i]) - int(id[i - 1])))

        elif (cy[(i - 1)] == 1 and cy[i] == 0):
            tcy.extend([0] * (int(id[i]) - int(id[i - 1])))

        elif (cy[(i - 1)] == 0 and cy[i] == 0):
            tcy.extend([0] * (int(id[i]) - int(id[i - 1])))

        elif (cy[(i - 1)] == 1 and cy[i] == 1):
            tcy.extend([1] * (int(id[i]) - int(id[i - 1])))

    maxx = max(max(mx), max(mn))

    if (tcy[(maxx - 1)] == 1):
        tcy.extend([0] * (len(data) - maxx))

    elif (tcy[(maxx - 1)] == 0):
        tcy.extend([1] * (len(data) - maxx))

    output = pd.DataFrame(tcy, columns=[data_type.value], index=date)
    print(f"data type: {data_type.value}")
    print(output)
    return output


#################################   OECD   直接定義高低   ######################


# 定義上升下降   前200 後30


cy_oecd = genSignal(
    DataType.OECD,
    data["oecd"],
    data['date'],
    interval_1=300,
    interval_2=30)

print(cy_oecd)

# oecdd.to_csv('/Users/doramaster/Desktop/python/final_cycle/cy_oecd.csv',sep=',',index = False,header=False)


#################################   crb1   直接定義高低 ########################


# 定義上升下降

cy_crb = genSignal(
    DataType.CRB,
    data['crb'],
    data['date'],
    interval_1=500,
    interval_2=30)

print(cy_crb)
# crb1.to_csv('/Users/doramaster/Desktop/python/final_cycle/cy_crb1.csv',sep=',',index = False,header=False)


#################################   us30r  採crb   ########################


# data = pd.read_csv('/Users/doramaster/Desktop/python/final_cycle/cy_index.csv')
# data["date"] = pd.to_datetime(data["date"])


# define KST 指標
def KST(df, k, d1, d2, d3, d4, r1, r2, r3, r4):
    M = df[k].diff(d1 - 1)
    N = df[k].shift(d1 - 1)
    ROC1 = M / N
    M = df[k].diff(d2 - 1)
    N = df[k].shift(d2 - 1)
    ROC2 = M / N
    M = df[k].diff(d3 - 1)
    N = df[k].shift(d3 - 1)
    ROC3 = M / N
    M = df[k].diff(d4 - 1)
    N = df[k].shift(d4 - 1)
    ROC4 = M / N
    KST = pd.Series(
        ROC1.rolling(r1).sum() +
        ROC2.rolling(r2).sum() *
        2 +
        ROC3.rolling(r3).sum() *
        3 +
        ROC4.rolling(r4).sum() *
        4,
        name='KST')
    df = df.join(KST)
    return df


df_usg = pd.DataFrame(data['us30r'])
kst_usg1 = KST(df_usg, 'us30r', 200, 250, 400, 500, 100, 100, 100, 200)['KST']


# 定義上升下降


cy_usg = genSignal(
    DataType.US30R,
    kst_usg1,
    data['date'],
    interval_1=200,
    interval_2=60)

# usgg.to_csv('/Users/doramaster/Desktop/python/final_cycle/cy_us30r.csv',sep=',',index = False,header=False)

print(cy_usg)


#################################   循環階段整合   #############################
all_index = pd.concat([cy_oecd, cy_crb, cy_usg], axis=1)
all_index.to_csv('./data/all_index.csv', sep=',')
print(all_index)


# data = all_index


cy_oecd = np.array(cy_oecd).flatten()
cy_crb = np.array(cy_crb).flatten()
cy_usg = np.array(cy_usg).flatten()

all_cy = []


for i in range(len(all_index)):

    if (cy_usg[i] == 0 and cy_oecd[i] == 0 and cy_crb[i] == 0):
        all_cy.extend([1])

    elif (cy_usg[i] == 0 and cy_oecd[i] == 0 and cy_crb[i] == 1):
        all_cy.extend([1])

    elif (cy_usg[i] == 0 and cy_oecd[i] == 1 and cy_crb[i] == 0):
        all_cy.extend([2])

    elif (cy_usg[i] == 0 and cy_oecd[i] == 1 and cy_crb[i] == 1):
        all_cy.extend([3])

    elif (cy_usg[i] == 1 and cy_oecd[i] == 1 and cy_crb[i] == 1):
        all_cy.extend([4])

    elif (cy_usg[i] == 1 and cy_oecd[i] == 1 and cy_crb[i] == 0):
        all_cy.extend([4])

    elif (cy_usg[i] == 1 and cy_oecd[i] == 0 and cy_crb[i] == 1):
        all_cy.extend([5])

    elif (cy_usg[i] == 1 and cy_oecd[i] == 0 and cy_crb[i] == 0):
        all_cy.extend([6])


data1 = pd.read_csv('./data/all_index.csv')
data1["date"] = pd.to_datetime(data1["date"])

all_cy = pd.DataFrame(all_cy, columns=['all_cy'], index=data1["date"])
all_cy = pd.concat([all_index, all_cy], axis=1)
all_cy.to_csv('./data/all_cy.csv', sep=',')

print(all_cy)
plt.plot(all_cy['all_cy'])
plt.show()
