

def get_distribute(max_x,len_distribute):
    x_all=0
    for i in range(len_distribute):
        x_all=x_all+max_x[i]
    distribute=[]
    for i in range(len_distribute):
        distribute.append(max_x[i]/x_all)
    return distribute