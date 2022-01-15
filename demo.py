import sys
# sys.path.append('../stl')
from stl.stlcg import *
import stl.stlviz as viz
from stl.stlcg import Expression
from stl.utils import print_learning_progress

import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib

t=np.arange(-3,3,0.2,dtype=np.float32)
# print(t)
x_np=0.5*np.exp(-t**2).reshape([1,t.shape[0],1])
w_np=(0.4*np.exp(-(t+0.5)**2)+.2*np.exp(-(t-3)**2)).reshape(1,-1)
# print(x_np.shape)
# print(w_np.shape)
w_np=w_np[:,:,None]
# print(w_np.shape)

x=torch.tensor(x_np,requires_grad=False)
w=torch.tensor(w_np,requires_grad=False)

c=torch.tensor(1.0,dtype=torch.float,requires_grad=True)
d=torch.tensor(0.9,dtype=torch.float,requires_grad=True)

# plt.figure(figsize=(8,5))
# plt.plot(t,x_np[0,:,0],'.-',markersize=15,label='x')
# plt.plot(t,w_np[0,:,0],'.-',markersize=15,label='w')
# plt.legend()
# plt.show()

e1=LessThan(lhs='x',val=c)
e2=GreaterThan(lhs='w',val=d)
e3=LessThan(lhs='w',val=d)
e4=Always(subformula=e1)
e5=Always(subformula=e3)

and1= Or(subformula1=e2,subformula2=e5)
# print(and1)
# viz.make_stl_graph(and1)

x_exp=Expression('x',x)
w_exp=Expression('w',w)
# print(x_exp)
c_exp=Expression('c',c)
d_exp=Expression('d',d)

e1=x_exp<=c_exp
e2=w_exp>=d_exp
e3=x_exp>d_exp
d4=w_exp<c_exp

phi_1=Always(subformula=e1)
phi_2=Always(subformula=e2)
phi_3=Eventually(subformula=e3)
# print(phi_3)

form=e1& e4
# print(form)

c=torch.tensor(555,dtype=torch.float,requires_grad=True)
x_exp=Expression('x',x)
w_exp=Expression('w',w)
# print(x_exp)
c_exp=Expression('c',c)
d_exp=Expression('d',d)

lt=x_exp<=c_exp
gt=w_exp<=d_exp
form=Always(lt)


var_dict={'c':c}
print(form)
inputs=x
lr=0.1
device=torch.device('cuda')
optimizer=torch.optim.Adam(var_dict.values(),lr=lr)
scale=0.5

# loss=form.robustness(inputs,scale=0.5).mean()**2
# print(loss.item())
for i in range(10000):
    sc=scale+i/500
    loss=form.robustness(inputs,scale=sc).mean()**2
    if i %100 ==0:
        print_learning_progress(form,inputs,var_dict,i,loss,sc)
    loss.backward()
    # print("loss:",loss.item())

    with torch.no_grad():
        c-=lr*c.grad
        c.grad.zero_()
# x_exp = Expression('x', x)
# w_exp = Expression('w', w)
# c_exp = Expression('c', c)
# d_exp = Expression('d', d)
# lt = x_exp <= c_exp
# gt = w_exp <= d_exp
# formula = Always(subformula=lt)
# inputs = x
# var_dict = {'c': c}
# print(formula)
# viz.make_stl_graph(formula)
# learning_rate = 0.05
# device = torch.device("cpu")
# optimizer = torch.optim.Adam(var_dict.values(), lr=learning_rate)
# scale = 0.5
# for i in range(20000):
#     sc = scale + i/500
#     loss = formula.robustness(inputs, scale=sc).mean()**2 #+ 0.001*(c**2 + d**2)
#     if i % 500 == 0:
#         print_learning_progress(formula, inputs, var_dict, i, loss, sc)
#     loss.backward()
#     with torch.no_grad():
#         c -= learning_rate * c.grad
#         c.grad.zero_()
