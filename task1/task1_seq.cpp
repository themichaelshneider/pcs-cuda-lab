#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Использование: ./sum_cpu <число запусков> <размер массива>\n";
        return 1;
    }

    int runs = std::atoi(argv[1]);
    int size = std::atoi(argv[2]);
    long long total_time = 0;

    for (int r = 0; r < runs; ++r) {
        int* arr = new int[size];
        for (int i = 0; i < size; ++i)
            arr[i] = rand() % 100;

        clock_t start = clock();

        long long sum = 0;
        for (int i = 0; i < size; ++i)
            sum += arr[i];

        clock_t end = clock();
        total_time += (end - start);

        delete[] arr;
    }

    double avg_time_ms = (double)total_time / runs / CLOCKS_PER_SEC;
    std::cout << "Среднее время выполнения: " << std::fixed << std::setprecision(8) << avg_time_ms << " с\n";

    return 0;
}
