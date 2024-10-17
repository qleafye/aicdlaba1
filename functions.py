import matplotlib.pyplot as plt
import numpy as np
from sorts import quicksort, shell_prap_sort, shell_sort, shell_hibb_sort, selection_sort, heap_sort, merge_sort, bubble_sort, insertion_sort
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def makeplot(_sizes, _time_values, name, comment):
    x = np.array(_sizes)
    y = np.array(_time_values)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(x.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    plt.title(f'Время выполнения сортировки {comment}')
    plt.xlabel('Размер массива')
    plt.ylabel('Время (секунды)')

    plt.scatter(x, y, c = "red")
    plt.plot(x, y_predicted, c = "blue", label = name)
    plt.legend()
    plt.grid(True)
    plt.show()

