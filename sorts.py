import random

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


def shell_hibb_sort(arr):
    n = len(arr)

    # Генерация последовательности Хиббарда
    gaps = []
    k = 1
    while True:
        gap = (2 ** k) - 1
        if gap >= n:
            break
        gaps.append(gap)
        k += 1

    # Сортировка по каждому шагу из последовательности Хиббарда
    for gap in reversed(gaps):
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp


def shell_prap_sort(arr):
    n = len(arr)

    # Генерация последовательности Пратта
    gaps = []
    i = 0
    while True:
        j = 0
        while True:
            gap = (2 ** i) * (3 ** j)
            if gap >= n:
                break
            gaps.append(gap)
            j += 1
        if (2 ** i) >= n:
            break
        i += 1

    # Удаляем дубликаты и сортируем шаги
    gaps = sorted(set(gaps))

    # Сортировка по каждому шагу из последовательности Пратта
    for gap in reversed(gaps):
        for k in range(gap, n):
            temp = arr[k]
            j = k
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]  # Выбираем опорный элемент
        left = [x for x in arr if x < pivot]  # Все элементы меньше опорного
        middle = [x for x in arr if x == pivot]  # Все элементы, равные опорному
        right = [x for x in arr if x > pivot]  # Все элементы больше опорного
        return quicksort(left) + middle + quicksort(right)


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