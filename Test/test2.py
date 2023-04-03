test = lambda x : x**2//(5+x)
test = lambda x : x//3+1
test1 = lambda x, y : y[:2] if int(x)//3 == 0 else y[-1]
# 1,2 는 [:2]
# 3,4,5 는 [-1]
a = [1,2,3]
for i in range(1,6):
    # print(test(i))
    print(test1(i,a))

