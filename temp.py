import torch
t = torch.LongTensor(4, 6).fill_(2)
t[0][3]=1
t[1][3]=1
t[0][2]=1
t[3][3]=1

#print(t[:-1, :])

a=torch.tensor([1,2,4 ,-1 ,15,  11, 9])
#print(a)
#print(a.max(0))
x=ord('\u200c')
print(x)
print (x == "\n")
print ('\u200c')