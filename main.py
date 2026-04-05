import numpy as np

# X = комнат, Метров
x_train = np.array([
    [2, 59],
    [2, 66],
    [1, 20],
    [3, 82],
    [3, 79],
    [2, 71],
    [2, 55],
    [3, 148],
    [3, 86],
    [1, 38],
    [1, 49],
    [1, 35]
])
# y = цена
y_train = np.array([7700000, 8700000, 3700000, 10700000, 8920000, 7750000, 6900000, 25180000, 8980000, 5280000, 5000000,
                    4100000])

# нормируем данные
x_norm = x_train/np.max(x_train, axis=0)
y_norm = y_train/np.max(y_train)

# J_wb
def cost_function(X, Y, W, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(W,X[i]) + b
        cost += pow(f_wb - Y[i],2)
    total_cost = cost/(2*m)
    return total_cost

def GradientComputing(X, Y, W, b):
    m = X.shape[0]
    dw = np.zeros(W.shape)
    db = 0
    for i in range(m):
        f_wb = np.dot(W, X[i]) + b
        db += f_wb - Y[i]
        dw += (f_wb - Y[i])*X[i]
    db = db/m
    dw = dw/m
    return dw, db


def GradientDescent(X, Y, W, b, alpha, iterations):
    epsilon = 1e-5
    dw, db = GradientComputing(X, Y, W, b)
    counter = 0
    while abs(db) > epsilon or np.linalg.norm(dw) > epsilon and counter < iterations:
        b -= alpha * db
        W -= alpha * dw
        if counter % 1000 == 0:
            print(f"Номер операции: {counter}")
            print(f"Ошибка: {cost_function(X, Y, W, b)}")
        dw, db = GradientComputing(X, Y, W, b)
        counter += 1
    return W, b, counter


def Prediction(x, X, Y, W_norm, b_norm):
    x_normed = x / np.max(X,axis=0)
    y_normed = np.dot(W_norm, x_normed) + b_norm
    y = y_normed * np.max(Y)
    return y


alpha_temp = 1
w_temp = np.zeros(x_norm.shape[1])
b_temp = 0
temp_iter = 300000

# данные для предсказания цены
square_meters  = 59
bedrooms = 2
example = np.array([bedrooms,square_meters])

final_W, final_b, op_count = GradientDescent(x_norm,y_norm,w_temp,b_temp,alpha_temp,temp_iter)

print(f"\nФинальная ошибка:{cost_function(x_norm, y_norm, final_W, final_b)}")
print(f"\nКоличество операций: {op_count}")
cost_of_apartment = int(Prediction(example, x_train, y_train, final_W, final_b))
print(f"\nПримерная Цена {bedrooms} комнатной квартиры в {square_meters} м^2: {cost_of_apartment:}₽.")