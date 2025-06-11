#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

void generateMatrix(int** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = rand() % 100 + 1;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "Использование: ./cpu_matrix_ops <запуски> <строки> <столбцы>\n";
        return 1;
    }

    int runs = atoi(argv[1]);
    int rows = atoi(argv[2]);
    int cols = atoi(argv[3]);
    int size = rows * cols;

    srand(time(nullptr));

    // Создание двумерных массивов
    int** A = new int*[rows];
    int** B = new int*[rows];
    int** add = new int*[rows];
    int** sub = new int*[rows];
    int** mul = new int*[rows];
    float** div = new float*[rows];

    for (int i = 0; i < rows; ++i) {
        A[i] = new int[cols];
        B[i] = new int[cols];
        add[i] = new int[cols];
        sub[i] = new int[cols];
        mul[i] = new int[cols];
        div[i] = new float[cols];
    }

    double t_add = 0.0, t_sub = 0.0, t_mul = 0.0, t_div = 0.0;

    for (int r = 0; r < runs; ++r) {
        generateMatrix(A, rows, cols);
        generateMatrix(B, rows, cols);

        clock_t start, end;

        // Сложение
        start = clock();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                add[i][j] = A[i][j] + B[i][j];
        end = clock();
        t_add += double(end - start) / CLOCKS_PER_SEC;

        // Вычитание
        start = clock();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                sub[i][j] = A[i][j] - B[i][j];
        end = clock();
        t_sub += double(end - start) / CLOCKS_PER_SEC;

        // Умножение
        start = clock();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                mul[i][j] = A[i][j] * B[i][j];
        end = clock();
        t_mul += double(end - start) / CLOCKS_PER_SEC;

        // Деление
        start = clock();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                div[i][j] = (float)A[i][j] / B[i][j];
        end = clock();
        t_div += double(end - start) / CLOCKS_PER_SEC;
    }

    cout << "\nСреднее время выполнения операций за " << runs << " запусков:\n";
    cout << "Сложение:   " << t_add / runs << " секунд\n";
    cout << "Вычитание:  " << t_sub / runs << " секунд\n";
    cout << "Умножение:  " << t_mul / runs << " секунд\n";
    cout << "Деление:    " << t_div / runs << " секунд\n";

    // Очистка памяти
    for (int i = 0; i < rows; ++i) {
        delete[] A[i];
        delete[] B[i];
        delete[] add[i];
        delete[] sub[i];
        delete[] mul[i];
        delete[] div[i];
    }
    delete[] A;
    delete[] B;
    delete[] add;
    delete[] sub;
    delete[] mul;
    delete[] div;

    return 0;
}
