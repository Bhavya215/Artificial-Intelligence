import numpy as np

X = [0.5,2.5]
Y = [0.2,0.9]

#sigmoid with parameters w&b
def f(w,b,x):
    return 1.0/(1.0 + np.exp(-(w*x+b)))

def error(w,b):                                #squared error
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5 * (fx - y)**2
    return err

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y) * (fx) * (1-fx)

def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y) * (fx) * (1-fx) * x

def do_gradient_descent():
    w,b,eta,max_epochs = -2,-2,1.0,1001        #eta is learning rate
    print("Epoch \t\t w \t\t\t b \t\t Error")
    print("---------------------------------------------------------------------------------")
    for i in range(max_epochs):
        dw,db= 0,0
        for x,y in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        w = w - eta * dw
        b = b - eta * db
        print(i,"\t",w,"\t",b,"\t",error(w,b))
    print("----------------------------------------------------------------------------------")
do_gradient_descent()