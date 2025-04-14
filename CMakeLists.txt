cmake_minimum_required(VERSION 3.10)
project(frobenius_mpi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === MS-MPI manual paths ===
set(MS_MPI_INCLUDE "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MS_MPI_LIB "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")  

include_directories(${MS_MPI_INCLUDE})
link_directories(${MS_MPI_LIB})

add_executable(frobenius_mpi
        main.cpp
        )

# Підключення бібліотеки MPI
include_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")
target_link_libraries(frobenius_mpi msmpi)

