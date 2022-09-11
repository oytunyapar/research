# 9761403244881934472
def bent_function_1(inp):
    return (inp[0] and inp[1]) ^ (inp[2] and inp[3]) ^ (inp[4] and inp[5])


# 10855359761657498816
def bent_function_2(inp):
    return (inp[0] and inp[1] and inp[2]) ^ (inp[1] and inp[3] and inp[4]) ^ (inp[0] and inp[1]) ^ (inp[0] and inp[3]) \
           ^ (inp[1] and inp[5]) ^ (inp[2] and inp[4]) ^ (inp[3] and inp[4])


# 7583035861148412416
def bent_function_3(inp):
    return (inp[0] and inp[1] and inp[2]) ^ (inp[0] and inp[3]) ^ (inp[1] and inp[4]) ^ (inp[2] and inp[5])


# 3745489178524800608
def bent_function_4(inp):
    return (inp[0] and inp[1] and inp[2]) ^ (inp[1] and inp[3] and inp[4]) ^ (inp[2] and inp[3] and inp[5]) ^ \
           (inp[0] and inp[3]) ^ (inp[1] and inp[5]) ^ (inp[2] and inp[3]) ^ (inp[2] and inp[4]) ^ (inp[2] and inp[5]) \
           ^ (inp[3] and inp[4]) ^ (inp[3] and inp[5])

