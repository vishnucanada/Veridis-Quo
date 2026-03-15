
def binary_search(lst, target):
    l = 0
    r = len(lst) - 1
    while l <= r:
        mid = (l + r) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] > target:
            r = mid - 1
        else:
            l = mid + 1
    return -1
lst = [1,2,3,4,5,6,7,8,9]
print(binary_search(lst, 10))
