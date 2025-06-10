#include <iostream>
#include <cstdlib>
#include <ctime>

void merge(int* arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int* L = new int[n1];
    int* R = new int[n2];

    for (int i = 0; i < n1; ++i) L[i] = arr[l + i];
    for (int j = 0; j < n2; ++j) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

void mergeSort(int* arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Использование: ./merge_sort <кол-во запусков> <размер массива>\n";
        return 1;
    }

    int runs = atoi(argv[1]);
    int size = atoi(argv[2]);

    srand(time(0));
    clock_t total = 0;

    for (int i = 0; i < runs; ++i) {
        int* arr = new int[size];
        for (int j = 0; j < size; ++j)
            arr[j] = rand();

        clock_t start = clock();
        mergeSort(arr, 0, size - 1);
        clock_t end = clock();

        total += (end - start);
        delete[] arr;
    }

    std::cout << "Среднее время последовательной сортировки: "
              << (double)total / runs / CLOCKS_PER_SEC << " секунд\n";

    return 0;
}
