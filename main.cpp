#include <iostream>
#include <cstdlib>   
#include <ctime>     
#include <cmath>     
#include <iomanip>   
#include <chrono>    
#include <mpi.h>     

// Зчитує параметри (кількість рядків, стовпців, чи виводити матрицю)
void parseArguments(int argc, char** argv, int& rows, int& cols, bool& show) {
    if (argc == 4) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
        char s = std::tolower(argv[3][0]);
        show = (s == 'y');
        std::cout << "rows=" << rows
                  << ", cols=" << cols
                  << ", show=" << (show ? "Y" : "N") << "\n\n";
    } else {
        std::cout << "Default values:\n";
        std::cout << "rows=" << rows << ", cols=" << cols
                  << ", show=" << (show ? "Y" : "N") << "\n\n";
    }
}

// Заповнює матрицю псевдовипадковими числами від 0 до 9
void fillMatrix(double* arr, int r, int c) {
    for (int i = 0; i < r * c; i++) {
        arr[i] = std::rand() % 10;
    }
}

// Виводить матрицю на екран
void printMatrix(const double* arr, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            std::cout << arr[i * c + j] << " ";
        }
        std::cout << "\n";
    }
}

// Обчислює локальну частину норми Фробеніуса 
double computeFrobeniusNormMPI(double* localBlock, int localRows, int cols) {
    // Локальна сума квадратів
    double localSum = 0.0;
    for (int i = 0; i < localRows * cols; i++) {
        double val = localBlock[i];
        localSum += val * val;
    }
    
    return localSum;
}

int main(int argc, char** argv) {
    // Параметри за замовчуванням
    int rowCount = 10, colCount = 10;
    bool showMatrixData = false;

    MPI_Init(&argc, &argv);

    int myRank = 0, totalProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);

    // Генеруємо матрицю тільки на ранзі 0
    double* matrix = nullptr;
    if (myRank == 0) {
        parseArguments(argc, argv, rowCount, colCount, showMatrixData);
        std::srand((unsigned)time(nullptr));
        matrix = new double[rowCount * colCount];
        fillMatrix(matrix, rowCount, colCount);
    }

    // Розсилаємо розміри
    MPI_Bcast(&rowCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Розраховуємо кількість рядків для кожного процесу
    int baseRows = rowCount / totalProcs;
    int extra = rowCount % totalProcs;
    int localRows = baseRows + ((myRank < extra) ? 1 : 0);

    // Буфер для локальної частини матриці
    double* localBlock = new double[localRows * colCount];

    // Змінні для Scatterv (тільки на ранзі 0)
    int* counts = nullptr;
    int* displs = nullptr;
    if (myRank == 0) {
        counts = new int[totalProcs];
        displs = new int[totalProcs];
        int offset = 0;
        for (int i = 0; i < totalProcs; i++) {
            int seg = baseRows + ((i < extra) ? 1 : 0);
            counts[i] = seg * colCount;
            displs[i] = offset;
            offset += counts[i];
        }
    }

    // Вимірюємо час
    double start = MPI_Wtime();
    
    // Розподіл даних по процесах через Scatterv
    MPI_Scatterv(matrix, counts, displs, MPI_DOUBLE,
                 localBlock, localRows * colCount, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Кожен процес обчислює свою локальну суму квадратів
    double localSum = computeFrobeniusNormMPI(localBlock, localRows, colCount);
    
    // Глобальна сума, підсумована на ранзі 0
    double globalSum = 0.0;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Обчислюємо кінцеву норму Фробеніуса (тільки ранг 0)
    double norm = 0.0;
    if (myRank == 0) {
        norm = std::sqrt(globalSum);
    }
    
    double end = MPI_Wtime();
    double elapsed = end - start;

    // Очищаємо ресурси
    delete[] localBlock;
    if (myRank == 0) {
        delete[] counts;
        delete[] displs;
    }

    // Виводимо результат (тільки 0-й процес)
    if (myRank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Frobenius Norm: " << norm << "\n";
        std::cout << "Elapsed Time:   " << elapsed << " s\n";

        if (showMatrixData) {
            std::cout << "\nMatrix Data:\n";
            printMatrix(matrix, rowCount, colCount);
        }
        delete[] matrix;
    }

    MPI_Finalize();
    return 0;
}