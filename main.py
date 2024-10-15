import random
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

    return arr

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

def shell_sort(arr): #последовательность шелла, хабара, прапа
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

def quicksort(nums):
    if len(nums) <= 1:
        return nums
    else:
        q = random.choice(nums)
        s_nums = []
        m_nums = []
        e_nums = []
        for n in nums:
            if n < q:
                s_nums.append(n)
            elif n > q:
                m_nums.append(n)
            else:
                e_nums.append(n)
        return quicksort(s_nums) + e_nums + quicksort(m_nums)


def insertion_sort(arr):
    # Проходим по всем элементам массива, начиная со второго
    for i in range(1, len(arr)):
        key = arr[i]  # Текущий элемент для вставки
        j = i - 1

        # Сдвигаем элементы массива, которые больше ключа, на одну позицию вправо
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        # Вставляем ключ на его правильное место
        arr[j + 1] = key
    return arr

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

sort_times = {'insertion': [], 'quick': [], 'shell': [], 'heap': [], 'bubble': [], 'merge': []}
'''
def makeplot(size, time, name, comment):
    plt.figure(figsize=(20, 6))
    plt.plot(size, time, marker='o', label=name)
    plt.title(f'Время выполнения сортировки {comment}')
    plt.xlabel('Размер массива')
    plt.ylabel('Время (секунды)')
    plt.xticks(size)
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

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






sort_times = {'insertion': [], 'quick': [], 'shell': [], 'heap': [], 'bubble': [], 'merge': []}



#для best
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    random_array.sort()
    sort_massives(random_array)

for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "best")

sort_times = {'insertion': [], 'quick': [], 'shell': [], 'heap': [], 'bubble': [], 'merge': []}

#для worst
for size in array_sizes:
    random_array = [random.randint(min_value, max_value) for _ in range(size)]
    random_array.sort(reverse=True)
    sort_massives(random_array)

for sort_name, times in sort_times.items():
    makeplot(array_sizes, times, sort_name, "worst")







