def productArray(nums):
    multiplier = 1
    output = [0] * len(nums)
    for i in range(len(nums)):
        output[i] *= multiplier
        multiplier *= nums[i]
    multipler = 1
    for i in range(len(nums) - 1, -1, -1):
        output[i] *= multiplier
        multiplier *= nums[i]
    return output
