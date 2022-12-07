def findLocal(arr,interval_1,interval_2):
    ## array length
    num = len(arr)

    ## empty lists for storage
    mx = []
    mn = []

    ## find local maxima and minima
    for i in range(interval_1 + 1, num - interval_2 + 1):
        ## local maxima
        if (max(arr[(i - interval_1):i]) < arr[i] > max(arr[(i + 1):(i + interval_2 + 1)])):
            mx.extend( [i + interval_2 + 1] )
        ## local minima
        elif (min(arr[(i - interval_1):i]) > arr[i] < min(arr[(i + 1):(i + interval_2 + 1)])):
            mn.extend([i + interval_2 + 1])


    return mx, mn
