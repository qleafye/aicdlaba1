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

def square(x):
    return x ** 2

def line(x):
    return x

def nlognsqaure(x):
    return x* np.log(x) ** 2

def nlogn(x):
    return x * np.log(x)

def n32(x):
    return x**(3/2)
def n54(x):
    return x**(5/4)
def n43(x):
    return x**(4/3)

x = np.linspace(1, 20001, 500)
'''
y = square(x)
plt.plot(x, y)
plt.title('Время выполнения с квадратичной ассимптоткой')
plt.xlabel('Размер массива')
plt.ylabel('Время (секунды)')
plt.show()

y = line(x)
plt.plot(x, y)
plt.title('Время выполнения с линейной ассимптоткой')
plt.xlabel('Размер массива')
plt.ylabel('Время (секунды)')
plt.show()

y = n32(x)
plt.plot(x, y)
plt.title('Время выполнения с n^3/2 ассимптоткой')
plt.xlabel('Размер массива')
plt.ylabel('Время (секунды)')
plt.show()

y = n43(x)
plt.plot(x, y)
plt.title('Время выполнения с n^4/3 ассимптоткой')
plt.xlabel('Размер массива')
plt.ylabel('Время (секунды)')
plt.show()

y = n54(x)
plt.plot(x, y)
plt.title('Время выполнения с n^5/4 ассимптоткой')
plt.xlabel('Размер массива')
plt.ylabel('Время (секунды)')
plt.show()

y = nlogn(x)
plt.plot(x, y)
plt.title('Время выполнения с nlogn ассимптоткой')
plt.xlabel('Размер массива')
plt.ylabel('Время (секунды)')
plt.show()

y = nlognsqaure(x)
plt.plot(x, y)
plt.title('Время выполнения с nlogn ^ 2 ассимптоткой')
plt.xlabel('Размер массива')
plt.ylabel('Время (секунды)')
plt.show()
'''

functions = [
    (square, 'O(n^2)', 'Время выполнения с квадратичной асимптоткой'),
    (line, 'O(n)', 'Время выполнения с линейной асимптоткой'),
    (n32, 'O(n^(3/2))', 'Время выполнения с n^(3/2) асимптоткой'),
    (n43, 'O(n^(4/3))', 'Время выполнения с n^(4/3) асимптоткой'),
    (n54, 'O(n^(5/4))', 'Время выполнения с n^(5/4) асимптоткой'),
    (nlogn, 'O(n log n)', 'Время выполнения с n log n асимптоткой'),
    (nlognsqaure, 'O(n log^2 n)', 'Время выполнения с n log^2 n асимптоткой')
]

# Проходим по каждой функции и строим отдельный график
for func, label, title in functions:
    plt.figure(figsize=(10, 6))

    # Вычисляем y значения
    y = func(x)

    plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel('Размер массива')
    plt.ylabel('Время (условные единицы)')
    #plt.ylim(0, max(y) * 1.1)  # Устанавливаем лимит по оси Y, чтобы не было обрезки
    plt.grid()
    plt.legend()
    plt.show()

plt.figure(figsize=(12, 8))

# Параметры графика
plt.ylim(0, 1e6)  # Ограничение по оси Y, чтобы лучше визуализировать
plt.xscale('log')  # Логарифмическая шкала по оси X

# Добавляем каждую функцию на график
plt.plot(x, square(x), label='O(n^2)')
plt.plot(x, line(x), label='O(n)')
plt.plot(x, n32(x), label='O(n^(3/2))')
plt.plot(x, n43(x), label='O(n^(4/3))')
plt.plot(x, n54(x), label='O(n^(5/4))')
plt.plot(x, nlogn(x), label='O(n log n)')
plt.plot(x, nlognsqaure(x), label='O(n log^2 n)')

# Настройка графика
plt.title('Ассимптотическая сложность различных алгоритмов сортировки')
plt.xlabel('Размер массива (логарифмическая шкала)')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid()
plt.show()



# Задаем параметры массива
array_sizes = []
for i in range(1,20002, 1000):
    array_sizes.append(i)  # размеры массивов для тестирования

sort_times = {'insertion': [], 'selection': [], 'quick': [], 'shell': [], 'shell_prap':[], "shell_hibb": [], 'heap': [], 'bubble': [], 'merge': []}

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
        selection_sort(random_array.copy())
        sort_times['selection'].append(time.time() - start_time)


        start_time = time.time()
        bubble_sort(random_array.copy())
        sort_times['bubble'].append(time.time() - start_time)


        start_time = time.time()
        merge_sort(random_array.copy())
        sort_times['merge'].append(time.time() - start_time)

plots_data = {sort_name: {'average': [], 'best': [], 'worst': [], 'almost': []} for sort_name in sort_times.keys()}

#для мид
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    sort_massives(random_array)

plots_data['insertion']['average'].append((sort_times['insertion']))
plots_data['quick']['average'].append((sort_times['quick']))
plots_data['shell']['average'].append((sort_times['shell']))
plots_data['shell_prap']['average'].append((sort_times['shell_prap']))
plots_data['shell_hibb']['average'].append((sort_times['shell_hibb']))
plots_data['heap']['average'].append((sort_times['heap']))
plots_data['bubble']['average'].append((sort_times['bubble']))
plots_data['merge']['average'].append((sort_times['merge']))
plots_data['selection']['average'].append((sort_times['selection']))


# Построение графика для всех средних
for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "mid")

# Построение общего графика для всех средних
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
plt.ylabel('Время (условные единицы)')
plt.xticks(array_sizes)
plt.grid(True)
plt.legend()
plt.show()






sort_times = {'insertion': [], 'selection': [], 'quick': [], 'shell': [], 'shell_prap':[], "shell_hibb": [], 'heap': [], 'bubble': [], 'merge': []}



#для best
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    random_array.sort()
    sort_massives(random_array)

plots_data['insertion']['best'].append((sort_times['insertion']))
plots_data['quick']['best'].append((sort_times['quick']))
plots_data['shell']['best'].append((sort_times['shell']))
plots_data['shell_prap']['best'].append((sort_times['shell_prap']))
plots_data['shell_hibb']['best'].append((sort_times['shell_hibb']))
plots_data['heap']['best'].append((sort_times['heap']))
plots_data['bubble']['best'].append((sort_times['bubble']))
plots_data['merge']['best'].append((sort_times['merge']))
plots_data['selection']['best'].append((sort_times['selection']))


for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "best")

sort_times = {'insertion': [], 'selection': [], 'quick': [], 'shell': [], 'shell_prap':[], "shell_hibb": [], 'heap': [], 'bubble': [], 'merge': []}

#для worst
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    random_array.sort(reverse=True)
    sort_massives(random_array)

plots_data['insertion']['worst'].append((sort_times['insertion']))
plots_data['quick']['worst'].append((sort_times['quick']))
plots_data['shell']['worst'].append((sort_times['shell']))
plots_data['shell_prap']['worst'].append((sort_times['shell_prap']))
plots_data['shell_hibb']['worst'].append((sort_times['shell_hibb']))
plots_data['heap']['worst'].append((sort_times['heap']))
plots_data['bubble']['worst'].append((sort_times['bubble']))
plots_data['merge']['worst'].append((sort_times['merge']))
plots_data['selection']['worst'].append((sort_times['selection']))


for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "worst")


sort_times = {'insertion': [], 'selection': [], 'quick': [], 'shell': [], 'shell_prap':[], "shell_hibb": [], 'heap': [], 'bubble': [], 'merge': []}
#для almost sort
ninetenmassive = []
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(int(size*0.9))]
    random_array.sort()
    random_array1 = [random.randint(min_value, max_value) for _ in range(int(size*0.1))]
    random_array1.sort(reverse=True)
    ninetenmassive = random_array + random_array1
    sort_massives(ninetenmassive)

plots_data['insertion']['almost'].append((sort_times['insertion']))
plots_data['quick']['almost'].append((sort_times['quick']))
plots_data['shell']['almost'].append((sort_times['shell']))
plots_data['shell_prap']['almost'].append((sort_times['shell_prap']))
plots_data['shell_hibb']['almost'].append((sort_times['shell_hibb']))
plots_data['heap']['almost'].append((sort_times['heap']))
plots_data['bubble']['almost'].append((sort_times['bubble']))
plots_data['merge']['almost'].append((sort_times['merge']))
plots_data['selection']['almost'].append((sort_times['selection']))


for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "90|10")


#делаем графики ^_^

print(plots_data['insertion'])
i = 0
for sort_name, sort_cases in plots_data.items():
    for sort_type, timings in sort_cases.items():
        print(f'Sort: {sort_name}, Case: {sort_type}', )
        for time in timings[0]:
            print("Размер массива:", array_sizes[i], "Время сортировки:", time)
            i += 1
        i = 0
        print(' ')


array_sizess = []
for i in range(1, 20002, 1000):
    array_sizess.append(i*0.1)
  # размеры массивов для тестирования

for sort_name, sort_cases in plots_data.items():
    plt.figure(figsize=(15, 6))
    for sort_type, timings in sort_cases.items():
        x = np.array(array_sizes)
        y = np.array(timings).flatten() # Берем данные для текущей сортировки и типа

        # Полиномиальная регрессия
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(x.reshape(-1, 1))
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y)
        y_predicted = poly_reg_model.predict(poly_features)

        # Построение графика

        plt.plot(x, y_predicted, label=f'{sort_name} - {sort_type}')




# Настройка графика
    plt.title('Время выполнения различных сортировок для разных случаев')
    plt.xlabel('Размер массива * 0.1')
    plt.ylabel('Время (секунды)')
    plt.xticks(array_sizes)
    plt.grid(True)
    plt.legend()
    plt.show()




