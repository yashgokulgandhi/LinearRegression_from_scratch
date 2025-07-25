import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Salary_Data.csv')


X = df["YearsExperience"].values
y = df["Salary"].values

def cost_funtion(X,y,w,b):
    cf=0
    m=len(X)
    for i in range(len(X)):
        f=w*X[i]+b
        mse=(f-y[i])**2
        cf+=mse
    
    return cf/(2*m)

def gradient_cost(X,y,w,b):
    dw=0
    db=0
    m=len(X)

    for i in range(m):
        wf=(w*X[i]+b-y[i])*X[i]
        bf = w*X[i]+b-y[i]
        dw+=wf
        db+=bf
    
    return dw/m,db/m


def gradient_decent(X,y,alpha,epochs):
    w=0
    b=0

    for i in range(epochs):
        dw,db=gradient_cost(X,y,w,b)
        w-=alpha*dw
        b-=alpha*db
        if i%100==0:
            print(f"w={w},b={b},cost_funtion={cost_funtion(X,y,w,b)}")
    
    return w,b



w,b=gradient_decent(X,y,0.01,1000)

plt.scatter(X, y, color='blue', label='Actual Data')
predicted_y = w * X + b
plt.plot(X, predicted_y, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.legend()
plt.grid(True)
plt.show()
