#include <iostream>
#include <chrono>
#include <cstdlib>

double run_seq(int N) {
    float* a = new float[N];
    for (int i=0; i<N; ++i) a[i] = 1.0f;
    auto st = std::chrono::high_resolution_clock::now();
    double sum = 0;
    for (int i=0; i<N; ++i) sum += a[i];
    auto en = std::chrono::high_resolution_clock::now();
    delete[] a;
    return std::chrono::duration<double, std::milli>(en-st).count();
}

int main(int argc, char* argv[]) {
    if (argc < 3) { std::cerr<<"Usage: ./seq num_runs N\n"; return 1; }
    int R = atoi(argv[1]), N = atoi(argv[2]);
    double total = 0;
    for (int i=0; i<R; ++i) total += run_seq(N);
    std::cout<<"Seq avg time: "<< (total/R) <<" ms\n";
    return 0;
}
