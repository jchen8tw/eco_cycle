def findLocal(arr,interval_1,interval_2):
    ## array length
    num = len(arr)

    ## empty lists for storage
    mx = []
    mn = []

    ## find local maxima and minima
    for i in range(interval_1 + 1, num - interval_2 + 1):
        ## local maxima
        if ((max(arr[(i - interval_1):i]) < arr[i] > max(arr[(i + 1):(i + interval_2 + 1)]))
            and (arr[i] > 1.05 * min(arr[(i - 100):(i)]))
            and (arr[i+interval_2 + 1] < 0.99*arr[i + 5])):
            mx.extend( [i + interval_2 + 1] )
        ## local minima
        elif ((min(arr[(i - interval_1):i]) > arr[i] < min(arr[(i + 1):(i + interval_2 + 1)]))
            and (arr[i] < 0.95*max(arr[(i - 200):( i )] ))
            and (arr[i+interval_2 + 1]> 1.01*arr[i + 5])):
            mn.extend([i + interval_2 + 1])


    return mx, mn
