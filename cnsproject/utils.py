def memo(f):
    m = {}
    def f_memo(x):
        if x in m:
            return m[x]
        else :
            m[x] = f(x)
            return m[x]
    
    return f_memo