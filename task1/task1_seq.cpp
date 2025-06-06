#include <iostream>
#include <cstdlib>
#include <time.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Использование: ./seq_sum <кол-во_запусков> <размер_массива>\n";
        return 1;
    }

    int n_runs = std::atoi(argv[1]);
    int size = std::atoi(argv[2]);

    float* array = new float[size];
    for (int i = 0; i < size; i++) {
        array[i] = 1.0f; // можно заменить на rand() / float(RAND_MAX)
    }

    double total_time = 0.0;
    float sum = 0.0f;

    for (int run = 0; run < n_runs; run++) {
        sum = 0.0f;

        clock_t start = clock();

        for (int i = 0; i < size; i++) {
            sum += array[i];
        }

        clock_t end = clock();
        total_time += 1000.0 * (end - start) / CLOCKS_PER_SEC; // в миллисекундах
    }

    std::cout << "Сумма элементов массива: " << sum << "\n";
    std::cout << "Среднее время выполнения: " << total_time / n_runs << " мс\n";

    delete[] array;
    return 0;
}
