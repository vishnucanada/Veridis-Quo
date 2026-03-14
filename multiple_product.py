def productItself(arr):
    multipler = 1
    output = [1] * len(arr)
    for i in range(len(arr)):
        output[i] *= multipler
        multipler *= arr[i]
    multiplier = 1
    for i in range(len(arr) - 1, -1, -1):
        output[i] *= multiplier
        multiplier *= arr[i]
    return output

nums = [1,2,3,4]
print(productItself(nums))
        