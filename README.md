
# BOSS Benchmarks

See [here](./Documentation/CodeStructure.md) for the code structure.

See [here](./Documentation/Specification.md) for the formal specification of the operators.


## Requirements

for compiling BOSS, MonetDB, DuckDB and benchmark code:

```
cmake >= 3.10
clang >= 9.0
libstdc++-dev >= 8.0
git
unzip
```

for generating missing data:
```
python3
pip
```

for Mathematica baseline:
```
install Wolfram Engine
authenticate to Wolfram
```

for Racket baseline:
```
Racket BC (CS racket has a different C API). Racket CS is the default from 8.0 onward so you need to compile it with the right flags
```

compatible (and tested) with:
* Linux Ubuntu 18.04 LTS (Bionic)
* Linux Ubuntu 20.04 LTS (Focal)
* Linux Debian 11 (Bullseye)

\+ compatible with most setup on MacOS (Clang) and Windows 10/11 (MSVC or WSL Clang) with custom adjustments to the instructions below.

## Instructions (for Linux Ubuntu/Debian)

### 1) installing dependencies (if required)

```
> sudo apt update
> sudo apt install cmake git unzip clang-9 libstdc++-8-dev
> sudo apt install python3 python3-pip
```

Note #1:  
Debian Bullseye provides only `libstdc++-9-dev` or `libstdc++-10-dev` which can be installed with an alternative command such as `apt install libstdc++-10-dev`.

Note #2:  
Installing the default `clang-10` on Ubuntu Focal or `clang-11` on Debian Bullseye with `apt install clang` is a working alternative, but the cmake command below need to be adjusted accordingly.

### 2) configuring and compiling project

```
> mkdir build
> cd build
> cmake -DCMAKE_INSTALL_PREFIX:PATH=.. -DCMAKE_C_COMPILER=clang-9   -DCMAKE_CXX_COMPILER=clang++-9 -DCMAKE_BUILD_TYPE=Release -B. ..
> cd ..
> cmake --build build --target install
```

to compile with Mathematica baseline support,
init the Mathematics CMake submodule with:
```
git submodule init
git submodule update
```
and add this flag to the cmake setup command:
```
-DBUILD_WOLFRAM_ENGINE=ON
```

### 3) Generating TPC-H dataset

(A) for all scale factors (0.001, 0.01, 0.1, 1, 2, 5, 10, 100):
```
> ./generate_tpch_data.sh
```

(B) for up to SF 1 only:
```
> ./generate_tpch_data.sh 0 4
```

### 4) Generating missing data for TPC-H

install python dependencies:
```
> pip install numpy pandas
```

(A) for all scale factors (0.001, 0.01, 0.1, 1, 2, 5, 10, 100):
```
> ./generate_missing_data.sh
```

(B) for up to SF 1 only:
```
> ./generate_missing_data.sh 0 4
```

### 5) Running the TPC-H benchmarks (without imputation)

with BOSS, MonetDB and DuckDB  
Queries Q1, Q3, Q6, Q9, Q18  
Scale factors 0.001, 0.01, 0.1, 1, 2, 5, 10, 100:

```
> cd bin
> LD_LIBRARY_PATH=../lib ./Benchmarks --library libBulkEngine.so --benchmark_filter="TPCH_Q"
```

### 6) Running the TPC-H benchmarks (with imputation)

with BOSS  
Queries Q1, Q3, Q6, Q9, Q18  
Scale factors 0.001, 0.01, 0.1, 1, 2, 5, 10, 100:

```
> cd bin
> LD_LIBRARY_PATH=../lib ./Benchmarks --library libBulkEngine.so --benchmark_filter="TPCH_I"
```

### 7) Running the CDC/FCC/ACS imputation benchmarks

with BOSS  
CDC dataset (queries Q1 to Q5):

```
> cd bin
> LD_LIBRARY_PATH=../lib ./Benchmarks --library libBulkEngine.so --benchmark_filter="CDC_I"
```

with BOSS  
FCC dataset (queries Q6 to Q9):

```
> cd bin
> LD_LIBRARY_PATH=../lib ./Benchmarks --library libBulkEngine.so --benchmark_filter="FCC_I"
```

with BOSS  
ACS dataset (column average):

```
> cd bin
> LD_LIBRARY_PATH=../lib ./Benchmarks --library libBulkEngine.so --benchmark_filter="ACS_I"
```

### 8) Running the TPC-H benchmarks with MonetDB baseline

Queries Q1, Q3, Q6, Q9, Q18  
Scale factors 0.001, 0.01, 0.1, 1, 2, 5, 10, 100:

```
> cd bin
> LD_LIBRARY_PATH=../lib ./Benchmarks --disable-indexes --benchmark_filter="TPCH_Q[0-9]+/MonetDB"
```

### 9) Running the TPC-H benchmarks with DuckDB baseline

Queries Q1, Q3, Q6, Q9, Q18  
Scale factors 0.001, 0.01, 0.1, 1, 2, 5, 10, 100:

```
> cd bin
> LD_LIBRARY_PATH=../lib ./Benchmarks --disable-indexes --benchmark_filter="TPCH_Q[0-9]+/DuckDB"
```

### 10) Running the TPC-H benchmarks with Mathematica baseline

Queries Q1, Q3, Q6, Q9, Q18  
Scale factors 0.001, 0.01, 0.1, 1, 2, 5, 10, 100:

```
> cd bin
> LD_LIBRARY_PATH=../lib ./Benchmarks --library libWolframEngine.so --benchmark_filter="TPCH_Q"
```

### 11) Running the TPC-H benchmarks with Racket baseline

Queries Q1, Q3, Q6, Q9, Q18  
Scale factor 1:

```
racket RacketBaseline/Engine.rkt data/tpch_1000MB RacketBaseline/Q1.rkt
racket RacketBaseline/Engine.rkt data/tpch_1000MB RacketBaseline/Q3.rkt
racket RacketBaseline/Engine.rkt data/tpch_1000MB RacketBaseline/Q6.rkt
racket RacketBaseline/Engine.rkt data/tpch_1000MB RacketBaseline/Q9.rkt
racket RacketBaseline/Engine.rkt data/tpch_1000MB RacketBaseline/Q18.rkt
```

### 12) Running the order preservation indexes benchmark

methods: PartitionIndexes, PartitionIndexesUnrolled, PartitionIndexesUnrolledAndCompressed, TwoPartitionIndexesUnrolled, GlobalIndex, CompressedGlobalIndex

Partition size: 1M

Number of partitions: 4, 16, 64

Zipf skew factors: 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8, 16

```
> cd bin
> ./MicroBenchmarks"
```

