def one_num(n):
    c = []
    for i in range(1,n+1):
        d = set()
        l = len(s:=str(i))
        if l > 1:
            for j in range(l-1):d.add(map(int,s[j:j+1]))   
            if len(d) == 1:c.append(i)
        else:c.append(i)
    print(len(c))
one_num(int(input()))