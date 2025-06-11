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

    double total_time_add = 0.0;
    double total_time_sub = 0.0;
    double total_time_mul = 0.0;
    double total_time_div = 0.0;

    for (int r = 0; r < runs; ++r) {
        generateArray(A, size);
        generateArray(B, size);

        clock_t start = clock();
        for (int i = 0; i < size; ++i) sum[i] = A[i] + B[i];
        clock_t end = clock();
        total_time_add += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        for (int i = 0; i < size; ++i) diff[i] = A[i] - B[i];
        end = clock();
        total_time_sub += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        for (int i = 0; i < size; ++i) prod[i] = A[i] * B[i];
        end = clock();
        total_time_mul += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        for (int i = 0; i < size; ++i) div[i] = (float)A[i] / B[i];
        end = clock();
        total_time_div += double(end - start) / CLOCKS_PER_SEC;
    }

    cout << "\nСреднее время за " << runs << " запусков:\n";
    cout << "Сложение:      " << total_time_add / runs << " секунд\n";
    cout << "Вычитание:     " << total_time_sub / runs << " секунд\n";
    cout << "Умножение:     " << total_time_mul / runs << " секунд\n";
    cout << "Деление:       " << total_time_div / runs << " секунд\n";

    delete[] A;
    delete[] B;
    delete[] sum;
    delete[] diff;
    delete[] prod;
    delete[] div;

    return 0;
}
