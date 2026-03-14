


flowers = [[1,6],[3,7],[9,12],[4,13]]
people = [2, 3, 7, 11]
output = [1, 2, 2, 2]

#brute force, for each pair compute valid timeframe

def binary_search(arr, target):
    l = 0 
    r = len(arr) - 1
    while l < r:
        mid = (l + r) // 2
        if arr[mid] == target:
def sol(flowers, people):
    index = 0
    output = [0] * len(people)
    while index < len(people):
        for flower in flowers:
            start, end = flower
            if start <= people[index] <= end:
                output[index] += 1

        index += 1
    return output

print(sol(flowers, people))
