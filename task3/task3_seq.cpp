// seq_operations.cpp
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

void generateArray(int* arr, int size) {
    for (int i = 0; i < size; ++i)
        arr[i] = rand() % 100 + 1;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Использование: ./seq_operations <кол-во_запусков> <размер_массива>\n";
        return 1;
    }

    int runs = atoi(argv[1]);
    int size = atoi(argv[2]);

    srand(time(nullptr));

    int* A = new int[size];
    int* B = new int[size];
    int* sum = new int[size];
    int* diff = new int[size];
    int* prod = new int[size];
    float* div = new float[size];

    double total_time = 0.0;

    for (int r = 0; r < runs; ++r) {
        generateArray(A, size);
        generateArray(B, size);

        clock_t start = clock();
        for (int i = 0; i < size; ++i) {
            sum[i] = A[i] + B[i];
            diff[i] = A[i] - B[i];
            prod[i] = A[i] * B[i];
            div[i] = (float)A[i] / B[i];
        }
        clock_t end = clock();

        double time_taken = double(end - start) / CLOCKS_PER_SEC;
        total_time += time_taken;

        cout << "Запуск " << r + 1 << ": Время выполнения = " << time_taken << " секунд\n";
    }

    cout << "Среднее время выполнения последовательной версии: " << total_time / runs << " секунд\n";

    delete[] A;
    delete[] B;
    delete[] sum;
    delete[] diff;
    delete[] prod;
    delete[] div;

    return 0;
}
