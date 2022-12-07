import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


data = pd.read_csv('/Users/doramaster/Desktop/python/cy_last/drCY/wave89.csv')
data["date"] = pd.to_datetime(data["date"])



index = data["TWA00"]

plt.figure(1)

plt.plot(data["date"], index, c = 'darkslategray', linewidth = 0.8, alpha = 0.8)
plt.title('台灣加權指數')



# cycle and inflation

cy = pd.read_csv('/Users/doramaster/Desktop/python/cy_last/drCY/dr_cy89.csv')
cypi = np.array(cy['cy'])



#cycle color

axis = [0]
color = []

for i in range( 1, (len(cypi) ) ):

    if ( cypi[i-1] != cypi[i] ):
        axis.extend( [(i+1)] )
        color.extend( [(cypi[i-1])] )

axis.extend( [len(cypi)] )
color.extend( [(cypi[len(cypi) - 1])])

print(axis)
print(color)


# final plot

datee = data["date"]


i=0
start = pd.Timestamp(datee[axis[i]  ])
end = pd.Timestamp(datee[axis[i + 1 ]])
int = np.linspace(start.value, end.value, (axis[i + 1] - axis[i] +1))
int = pd.to_datetime(int)

if  color[0] == 1 :
        plt.fill_between( int, index[0 : (axis[1] - axis[0] + 1)], color = 'darkorange',alpha = 0.7 )

elif color[0] == 2 :
        plt.fill_between( int, index[0 : (axis[1] - axis[0] + 1)], color = 'gray',alpha = 0.7 )

elif color[0] == 3 :
        plt.fill_between( int, index[0 : (axis[1] - axis[0] + 1)], color = 'yellow',alpha = 0.7 )

elif color[0] == 4 :
        plt.fill_between( int, index[0 : (axis[1] - axis[0] + 1)], color = 'dodgerblue',alpha = 0.7 )

elif color[0] == 5 :
        plt.fill_between( int, index[0 : (axis[1] - axis[0] + 1)], color = 'limegreen',alpha = 0.7 )

elif color[0] == 6 :
        plt.fill_between( int, index[0 : (axis[1] - axis[0] + 1)], color = 'mediumorchid',alpha = 0.7 )


i=1
start = pd.Timestamp(datee[axis[i]  ])
end = pd.Timestamp(datee[axis[i + 1 ]])
int = np.linspace(start.value, end.value, (axis[i + 1] - axis[i] ))
int = pd.to_datetime(int)

if  color[i] == 1 :
        plt.fill_between( int, index[(axis[1] - axis[0] +1) : (axis[2] +1)], color = 'darkorange',alpha = 0.7 )

elif color[i] == 2 :
    plt.fill_between( int, index[(axis[1] - axis[0] +1) : (axis[2] +1)], color = 'gray',alpha = 0.7 )

elif color[i] == 3 :
        plt.fill_between( int, index[(axis[1] - axis[0] +1) : (axis[2] +1)], color = 'yellow',alpha = 0.7 )

elif color[i] == 4 :
        plt.fill_between( int, index[(axis[1] - axis[0] +1) : (axis[2] +1)], color = 'dodgerblue',alpha = 0.7 )

elif color[i] == 5 :
        plt.fill_between( int, index[(axis[1] - axis[0] +1) : (axis[2] +1)], color = 'limegreen',alpha = 0.7 )

elif color[i] == 6 :
        plt.fill_between( int, index[(axis[1] - axis[0] +1) : (axis[2] +1)], color = 'mediumorchid',alpha = 0.7 )



for i in range( 2, (len(color) -1 ) ):

    start = pd.Timestamp(datee[axis[i] ])
    end = pd.Timestamp(datee[axis[i + 1 ]])
    int = np.linspace(start.value, end.value, (axis[i + 1] - axis[i] ))
    int = pd.to_datetime(int)

    if color[i] == 1:
        plt.fill_between(int, index[(axis[i] +1) : (axis[i+1] +1)], color='darkorange', alpha=0.7)

    elif color[i] == 2:
        plt.fill_between(int, index[(axis[i] +1) : (axis[i+1] +1)], color='gray', alpha=0.7)

    elif color[i] == 3:
        plt.fill_between(int, index[(axis[i] +1) : (axis[i+1] +1)], color='yellow', alpha=0.7)

    elif color[i] == 4:
        plt.fill_between(int, index[(axis[i] +1) : (axis[i+1] +1)], color='dodgerblue', alpha=0.7)

    elif color[i] == 5:
        plt.fill_between(int, index[(axis[i] +1) : (axis[i+1] +1)], color='limegreen', alpha=0.7)

    elif color[i] == 6:
        plt.fill_between(int, index[(axis[i] +1) : (axis[i+1] +1)], color='mediumorchid', alpha=0.7)






i = len(color)

start = pd.Timestamp(datee[axis[i-1]  ])
end = pd.Timestamp(datee[ axis[i] -1 ])
int = np.linspace(start.value, end.value, (axis[i] - axis[i-1] -1 ))
int = pd.to_datetime(int)

if color[i - 1] == 1:
    plt.fill_between(int, index[(axis[i-1] + 1): (axis[i] )], color='darkorange', alpha=0.7)

elif color[i -1 ] == 2:
    plt.fill_between(int, index[(axis[i-1] + 1): (axis[i] )], color='gray', alpha=0.7)

elif color[i - 1] == 3:
    plt.fill_between(int, index[(axis[i-1] + 1): (axis[i] )], color='yellow', alpha=0.7)

elif color[i - 1] == 4:
    plt.fill_between(int, index[(axis[i-1] + 1): (axis[i] )], color='dodgerblue', alpha=0.7)

elif color[i - 1] == 5:
    plt.fill_between(int, index[(axis[i-1] + 1): (axis[i] )], color='limegreen', alpha=0.7)

elif color[i - 1] == 6:
    plt.fill_between(int, index[(axis[i-1] + 1): (axis[i])], color='mediumorchid', alpha=0.7)

plt.ylim([(0.95*min(index)), (1.05*max(index))])



color1 = ['darkorange','gray', 'yellow', 'dodgerblue', 'limegreen', 'mediumorchid']
labels = ['經濟收縮-階段1','經濟收縮-階段2','經濟收縮-階段3','經濟擴張-階段4','經濟擴張-階段5','經濟擴張-階段6']
patches = [mpatches.Patch(color = color1[i], label = "{:s}".format(labels[i])) for i in range(len(color1))]
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height* 0.95])
ax.legend(handles = patches,loc = 2, ncol=2)


## cycle line
ff = []
for i in range( 50, (len(cypi)-1 ) ):

    if ( cypi[i] != cypi[i+1] and sum(cypi[(i-10):i])/10 == cypi[i] and
            cypi[i+1] == cypi[i+2]):
        ff.extend( [i+1] )

turn_date = pd.DataFrame(data["date"][ff])

turn_date=turn_date.reset_index()

for i in range( 0 , len(turn_date['date'])):
    plt.axvline(x = turn_date['date'][i], ls='--', c='red', linewidth = 0.7)

plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/台灣加權指數.jpg', dpi=300)

print(color)
print(turn_date)
print(data['TWA00'][235])


### 找斷點階段

cy1 = []
cy2 = []
cy3 = []
cy4 = []
cy5 = []
cy6 = []

for i in range( 1 , len(color)):
    if color[i] == 1 : cy1.append(turn_date['index'][i-1])
    elif color[i] == 2 : cy2.append(turn_date['index'][i-1])
    elif color[i] == 3: cy3.append(turn_date['index'][i-1])
    elif color[i] == 4: cy4.append(turn_date['index'][i-1])
    elif color[i] == 5: cy5.append(turn_date['index'][i-1])
    elif color[i] == 6: cy6.append(turn_date['index'][i-1])


print(cy6)

### 短天期報酬

def ret(cycle):
    short_retrun1 = []
    short_retrun2 = []
    short_retrun3 = []
    short_retrun4 = []
    short_retrun5 = []
    short_retrun6 = []

    for i in range(0, len(cycle)):
        x = cycle[i]
        ret1 = (data['TWA00'][x + 20] - data['TWA00'][x]) / data['TWA00'][x]
        ret2 = (data['TWB23'][x + 20] - data['TWB23'][x]) / data['TWB23'][x]
        ret3 = (data['TWB28'][x + 20] - data['TWB28'][x]) / data['TWB28'][x]
        ret4 = (data['TWA06'][x + 20] - data['TWA06'][x]) / data['TWA06'][x]
        ret5 = (data['OTC'][x + 20] - data['OTC'][x]) / data['OTC'][x]
        ret6 = (data['AN'][x + 20] - data['AN'][x]) / data['AN'][x]
        short_retrun1.append(ret1)
        short_retrun2.append(ret2)
        short_retrun3.append(ret3)
        short_retrun4.append(ret4)
        short_retrun5.append(ret5)
        short_retrun6.append(ret6)

    short_retrun1 = [x for x in short_retrun1 if np.isnan(x) == False]
    short_retrun2 = [x for x in short_retrun2 if np.isnan(x) == False]
    short_retrun3 = [x for x in short_retrun3 if np.isnan(x) == False]
    short_retrun4 = [x for x in short_retrun4 if np.isnan(x) == False]
    short_retrun5 = [x for x in short_retrun5 if np.isnan(x) == False]
    short_retrun6 = [x for x in short_retrun6 if np.isnan(x) == False]

    m1 = 100*sum(short_retrun1)/len(short_retrun1)
    m2 = 100*sum(short_retrun2)/len(short_retrun2)
    m3 = 100*sum(short_retrun3)/len(short_retrun3)
    m4 = 100*sum(short_retrun4)/len(short_retrun4)
    m5 = 100*sum(short_retrun5)/len(short_retrun5)
    m6 = 100*sum(short_retrun6)/len(short_retrun6)

    each_m = [m1,m2,m3,m4,m5,m6]
    return  each_m

re_cy1 = ret(cy1)
re_cy2 = ret(cy2)
re_cy3 = ret(cy3)
re_cy4 = ret(cy4)
re_cy5 = ret(cy5)



def ret1(cycle):
    short_retrun1 = []
    short_retrun2 = []
    short_retrun3 = []
    short_retrun4 = []
    short_retrun5 = []
    short_retrun6 = []

    for i in range(0, len(cycle)-1):
        x = cycle[i]
        ret1 = (data['TWA00'][x + 20] - data['TWA00'][x]) / data['TWA00'][x]
        ret2 = (data['TWB23'][x + 20] - data['TWB23'][x]) / data['TWB23'][x]
        ret3 = (data['TWB28'][x + 20] - data['TWB28'][x]) / data['TWB28'][x]
        ret4 = (data['TWA06'][x + 20] - data['TWA06'][x]) / data['TWA06'][x]
        ret5 = (data['OTC'][x + 20] - data['OTC'][x]) / data['OTC'][x]
        ret6 = (data['AN'][x + 20] - data['AN'][x]) / data['AN'][x]
        short_retrun1.append(ret1)
        short_retrun2.append(ret2)
        short_retrun3.append(ret3)
        short_retrun4.append(ret4)
        short_retrun5.append(ret5)
        short_retrun6.append(ret6)


    short_retrun1 = [x for x in short_retrun1 if np.isnan(x) == False]
    short_retrun2 = [x for x in short_retrun2 if np.isnan(x) == False]
    short_retrun3 = [x for x in short_retrun3 if np.isnan(x) == False]
    short_retrun4 = [x for x in short_retrun4 if np.isnan(x) == False]
    short_retrun5 = [x for x in short_retrun5 if np.isnan(x) == False]
    short_retrun6 = [x for x in short_retrun6 if np.isnan(x) == False]
    m1 = 100*sum(short_retrun1)/len(short_retrun1)
    m2 = 100*sum(short_retrun2)/len(short_retrun2)
    m3 = 100*sum(short_retrun3)/len(short_retrun3)
    m4 = 100*sum(short_retrun4)/len(short_retrun4)
    m5 = 100*sum(short_retrun5)/len(short_retrun5)
    m6 = 100*sum(short_retrun6)/len(short_retrun6)

    each_m = [m1,m2,m3,m4,m5,m6]
    return  each_m


re_cy6 = ret1(cy6)

###短天期 各分類報酬變動
cat = []
for i in range( 0 , 6):
    cat.append( [re_cy1[i], re_cy2[i] ,re_cy3[i] ,re_cy4[i] ,re_cy5[i] ,re_cy6[i]])


df_cat = pd.DataFrame(cat).T
df_cat.to_csv('/Users/doramaster/Desktop/python/cy_last/drCY/博士短期各階段報酬.csv')
print(df_cat)


step = ['cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'cycle6']

plt.figure(2)
plt.plot(cat[0], c = 'dodgerblue', linewidth = 2, alpha = 0.8, ls='--', marker='o', mec='r', mfc='r')
x = np.arange(len(cat[0]))
plt.xticks(x, step)
ax2 = plt.gca()
ax2.grid(color='gray', linestyle='--', linewidth=0.8,alpha=0.5)
plt.title('短線-各期台灣加權指數平均報酬')
plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-各期台灣加權指數平均報酬.jpg', dpi=300)

plt.figure(3)
plt.plot(cat[1], c = 'dodgerblue', linewidth = 2, alpha = 0.8, ls='--', marker='o', mec='r', mfc='r')
x = np.arange(len(cat[1]))
plt.xticks(x, step)
ax3 = plt.gca()
ax3.grid(color='gray', linestyle='--', linewidth=0.8,alpha=0.5)
plt.title('短線-各期電子類指數平均報酬')
plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-各期電子類指數平均報酬.jpg', dpi=300)

plt.figure(4)
plt.plot(cat[2], c = 'dodgerblue', linewidth = 2, alpha = 0.8, ls='--', marker='o', mec='r', mfc='r')
x = np.arange(len(cat[2]))
plt.xticks(x, step)
ax4 = plt.gca()
ax4.grid(color='gray', linestyle='--', linewidth=0.8,alpha=0.5)
plt.title('短線-各期金融保險平均報酬')
plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-各期金融保險平均報酬.jpg', dpi=300)

plt.figure(5)
plt.plot(cat[3], c = 'dodgerblue', linewidth = 2, alpha = 0.8, ls='--', marker='o', mec='r', mfc='r')
x = np.arange(len(cat[3]))
plt.xticks(x, step)
ax5 = plt.gca()
ax5.grid(color='gray', linestyle='--', linewidth=0.8,alpha=0.5)
plt.title('短線-各期非金電指數平均報酬')
plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-各期非金電指數平均報酬.jpg', dpi=300)

plt.figure(6)
plt.plot(cat[4], c = 'dodgerblue', linewidth = 2, alpha = 0.8, ls='--', marker='o', mec='r', mfc='r')
x = np.arange(len(cat[4]))
plt.xticks(x, step)
ax6 = plt.gca()
ax6.grid(color='gray', linestyle='--', linewidth=0.8,alpha=0.5)
plt.title('短線-各期OTC指數平均報酬')
plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-各期OTC指數平均報酬.jpg', dpi=300)

plt.figure(7)
plt.plot(cat[5], c = 'dodgerblue', linewidth = 2, alpha = 0.8, ls='--', marker='o', mec='r', mfc='r')
x = np.arange(len(cat[5]))
plt.xticks(x, step)
ax7 = plt.gca()
ax7.grid(color='gray', linestyle='--', linewidth=0.8,alpha=0.5)
plt.title('短線-各期台灣安聯大壩平均報酬')
plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-各期台灣安聯大壩平均報酬.jpg', dpi=300)



### 短天期-平均報酬

plt.figure(8)
cat = ['加權指數','電子類指數','金融保險指數','非金電指數'
        ,'OTC指數','台灣安聯大壩']
mn = re_cy1
x = np.arange(len(cat))
plt.bar(x, mn, color=['dodgerblue'],alpha=0.5)
plt.xticks(x, cat)
plt.ylabel('return(%)')
plt.title('短線-第一階段各指數報酬')

plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-第一階段各指數報酬.jpg', dpi=300)

plt.figure(9)
cat = ['加權指數','電子類指數','金融保險指數','非金電指數'
        ,'OTC指數','台灣安聯大壩']
mn = re_cy2
x = np.arange(len(cat))
plt.bar(x, mn, color=['dodgerblue'],alpha=0.5)
plt.xticks(x, cat)
plt.ylabel('return(%)')
plt.title('短線-第二階段各指數報酬')

plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-第二階段各指數報酬.jpg', dpi=300)

plt.figure(10)
cat = ['加權指數','電子類指數','金融保險指數','非金電指數'
        ,'OTC指數','台灣安聯大壩']
mn = re_cy3
x = np.arange(len(cat))
plt.bar(x, mn, color=['dodgerblue'],alpha=0.5)
plt.xticks(x, cat)
plt.ylabel('return(%)')
plt.title('短線-第三階段各指數報酬')

plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-第三階段各指數報酬.jpg', dpi=300)

plt.figure(11)
cat = ['加權指數','電子類指數','金融保險指數','非金電指數'
        ,'OTC指數','台灣安聯大壩']
mn = re_cy4
x = np.arange(len(cat))
plt.bar(x, mn, color=['dodgerblue'],alpha=0.5)
plt.xticks(x, cat)
plt.ylabel('return(%)')
plt.title('短線-第四階段各指數報酬')

plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-第四階段各指數報酬.jpg', dpi=300)

plt.figure(12)
cat = ['加權指數','電子類指數','金融保險指數','非金電指數'
        ,'OTC指數','台灣安聯大壩']
mn = re_cy5
x = np.arange(len(cat))
plt.bar(x, mn, color=['dodgerblue'],alpha=0.5)
plt.xticks(x, cat)
plt.ylabel('return(%)')
plt.title('短線-第五階段各指數報酬')

plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-第五階段各指數報酬.jpg', dpi=300)

plt.figure(13)
cat = ['加權指數','電子類指數','金融保險指數','非金電指數'
        ,'OTC指數','台灣安聯大壩']
mn = re_cy6
x = np.arange(len(cat))
plt.bar(x, mn, color=['dodgerblue'],alpha=0.5)
plt.xticks(x, cat)
plt.ylabel('return(%)')
plt.title('短線-第六階段各指數報酬')

plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/短線-第六階段各指數報酬.jpg', dpi=300)

print(re_cy1)
print(re_cy2)
print(re_cy3)
print(re_cy4)
print(re_cy5)
print(re_cy6)




##------------------------------------------------------

###短天期 各分類報酬變動
cat = []
for i in range( 0 , 6):
    cat.append( [re_cy1[i], re_cy2[i] ,re_cy3[i] ,re_cy4[i] ,re_cy5[i] ,re_cy6[i]])


df_cat = pd.DataFrame(cat).T
df_cat.to_csv('/Users/doramaster/Desktop/python/cy_last/drCY/博士短期各階段報酬.csv')
print(df_cat)


step = ['cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'cycle6']

plt.figure(figsize=(7, 6))
plt.plot(cat[0], c = 'r', linewidth = 2, alpha = 0.8, ls='-', marker='o', mec='r', mfc='r', label = '台灣指數')
x = np.arange(len(cat[0]))
plt.xticks(x, step)

plt.plot(cat[1], c = 'blue', linewidth = 2, alpha = 0.6, ls='-', marker='o', mec='blue', mfc='blue', label ='電子')
x = np.arange(len(cat[1]))
plt.xticks(x, step)

plt.plot(cat[2], c = 'green', linewidth = 2, alpha = 0.6, ls='-', marker='o', mec='green', mfc='green', label ='金融保險')
x = np.arange(len(cat[2]))
plt.xticks(x, step)

plt.plot(cat[3], c = 'goldenrod', linewidth = 2, alpha = 0.6, ls='-', marker='o', mec='goldenrod', mfc='goldenrod', label ='非金電')
x = np.arange(len(cat[3]))
plt.xticks(x, step)

plt.plot(cat[4], c = 'darkviolet', linewidth = 2, alpha = 0.8, ls='-', marker='o', mec='darkviolet', mfc='darkviolet', label ='OTC')
x = np.arange(len(cat[4]))
plt.xticks(x, step)

ax2 = plt.gca()
ax2.grid(color='gray', linestyle='--', linewidth=0.8,alpha=0.5)
plt.legend()
plt.title('短線-全部各期平均報酬')
plt.savefig('/Users/doramaster/Desktop/python/cy_last/drCY/19短線-全部各期平均報酬.jpg', dpi=300)


plt.show()





