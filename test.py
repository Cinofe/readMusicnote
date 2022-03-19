h, m = map(int,input().split())
t = int(input())
m += t%60
if m >= 60:
    m = 0
    h += 1
h += t//60
if h >= 24:
    h = 0
print(h,m,sep=' ')
    