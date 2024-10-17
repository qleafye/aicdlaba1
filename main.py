import random
import time
from sorts import *
import matplotlib.pyplot as plt
import numpy as np
from functions import makeplot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# == з е р н о == tri topora ==
random.seed(777)
'''
# Задаем параметры массива
array_size = 10000  # размер массива
min_value = 1  # минимальное значение
max_value = 1000000  # максимальное значение

# Заполняем массив случайными числами
a = []
b = []
random_array = [random.randint(min_value, max_value) for _ in range(array_size)]

start_time = time.time()
a = insertion_sort(random_array)
end_time = time.time()
print("insertion", end_time - start_time)

start_time = time.time()
b = quicksort(random_array)
end_time = time.time()
print("quick", end_time - start_time)

start_time = time.time()
b = shell_sort(random_array)
end_time = time.time()
print("shell", end_time - start_time)

start_time = time.time()
b = heap_sort(random_array)
end_time = time.time()
print("heap", end_time - start_time)

start_time = time.time()
b = bubble_sort(random_array)
end_time = time.time()
print("bubble", end_time - start_time)

start_time = time.time()
b = merge_sort(random_array)
end_time = time.time()
print("merge", end_time - start_time)
'''

# Задаем параметры массива
array_sizes = []
for i in range(1,10001, 1000):
    array_sizes.append(i)  # размеры массивов для тестирования

sort_times = {'insertion': [], 'quick': [], 'shell': [], 'shell_prap':[], "shell_hibb": [], 'heap': [], 'bubble': [], 'merge': []}

min_value = 1
max_value = 1000000

def sort_massives(random_array):
        start_time = time.time()
        insertion_sort(random_array.copy())
        sort_times['insertion'].append(time.time() - start_time)


        start_time = time.time()
        quicksort(random_array.copy())
        sort_times['quick'].append(time.time() - start_time)


        start_time = time.time()
        shell_sort(random_array.copy())
        sort_times['shell'].append(time.time() - start_time)

        start_time = time.time()
        shell_hibb_sort(random_array.copy())
        sort_times['shell_hibb'].append(time.time() - start_time)

        start_time = time.time()
        shell_prap_sort(random_array.copy())
        sort_times['shell_prap'].append(time.time() - start_time)


        start_time = time.time()
        heap_sort(random_array.copy())
        sort_times['heap'].append(time.time() - start_time)


        start_time = time.time()
        bubble_sort(random_array.copy())
        sort_times['bubble'].append(time.time() - start_time)


        start_time = time.time()
        merge_sort(random_array.copy())
        sort_times['merge'].append(time.time() - start_time)

#для мид
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    sort_massives(random_array)



# Построение графика для всех средних
for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "mid")

# Построение общего графика для всех сортировок
plt.figure(figsize=(20, 6))
for sort_name, times in sort_times.items():
    x = np.array(array_sizes)
    y = np.array(times)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(x.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)
    plt.plot(x, y_predicted , label = sort_name)

plt.title('Время выполнения различных сортировок для среднего случая')
plt.xlabel('Размер массива')
plt.ylabel('Время (секунды)')
plt.xticks(array_sizes)
plt.grid(True)
plt.legend()
plt.show()






sort_times = {'insertion': [], 'quick': [], 'shell': [], 'shell_prap':[], "shell_hibb": [], 'heap': [], 'bubble': [], 'merge': []}



#для best
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    random_array.sort()
    sort_massives(random_array)

for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "best")

sort_times = {'insertion': [], 'quick': [], 'shell': [], 'shell_prap':[], "shell_hibb": [], 'heap': [], 'bubble': [], 'merge': []}

#для worst
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    random_array.sort(reverse=True)
    sort_massives(random_array)

for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "worst")

sort_times = {'insertion': [], 'quick': [], 'shell': [], 'shell_prap':[], "shell_hibb": [], 'heap': [], 'bubble': [], 'merge': []}
#для almost sort
ninetenmassive = []
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(int(size*0.9))]
    random_array.sort()
    random_array1 = [random.randint(min_value, max_value) for _ in range(int(size*0.1))]
    random_array1.sort(reverse=True)
    ninetenmassive = random_array + random_array1
    sort_massives(ninetenmassive)

for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "90|10")







