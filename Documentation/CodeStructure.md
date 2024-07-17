
# BOSS Core

`Core/` is the sub-project containing the code for the core BOSS Expression API.
The most important files are:
* `Core/Source/Expression.hpp`: C++ interface for the Expression API
* `Core/Source/Shims/Expression.rkt`: LISP Scheme interface for the Expression API
* `Core/Source/Engine.hpp`: base class for implementing an engine for BOSS (BulkEngine and WolframEngine inherits from that)

# Bulk Engine

`BulkEngine/` is the sub-project containing the code for the Bulk engine, i.e. the BOSS implementation of an homoiconic data processing engine.
The most important files are:
* `BulkEngine/Source/Bulk.cpp` : BOSS processing engine
	* `Engine::evaluate` is the entry point to evaluate an expression
	* `Engine::evaluateAndConsumeArguments` for the argument evaluation logic
	* `Engine::evaluateSymbols` for the push-driven processing logic
* `BulkEngine/Source/ArrowExtensions/ComplexExpressionBuilder.hpp`
	* `appendExpression` for the Shape-Wise Partitioning & Decomposition logic
	* `isMatchingType` for shape comparison
* `BulkEngine/Source/Table.hpp`: higher-level partition handling logic
* `BulkEngine/Source/RelationalOperators.hpp`: implementation of the relational operators
	* l.280: Select
	* l.446: Project
	* l.767: Group
	* l.1854: Sort
	* l.2032: Top
	* l.2206: Join
* `BulkEngine/Source/ImputationOperators.hpp`: implementation of the imputation operators
	* l.121: Decision Tree
	* l.551: HotDeck
	* l.679: Approximate Mean
* `BulkEngine/Source/BulkOperators.hpp`: implementation of other bulk operators
	* l.1381: Evaluate

# Wolfram baseline

`WolframEngine/` is the sub-project containing the code for the Wolfram baseline. The implementation is contained in a single file: `WolframEngine/Source/WolframEngine.cpp`.

# Racket baseline

`RacketBaseline/` is the sub-project containing the code for the Racket baseline. The implementation is contained in a single file: `RacketBaseline/Engine.rkt`.

# BOSS Benchmarks

The macro benchmarks are all implemented in the file `Benchmarks/Benchmarks.cpp'.
The queries are in `BOSSQueries()`:
* lines 1130-1329 for TPC-H with BOSS's own query plans
* lines 1332-1446 for TPC-H with the query plans variants using DuckDB plans
* lines 1449 onwards for TPC-H, CDC, FCC, ACS queries with imputation

# MonetDB Benchmarks

MonetDB integration code is in `Benchmarks/MonetDB.hpp`.
The queries are in `Benchmarks/Benchmarks.cpp`, `monetdbQueries()`:
* lines 2072-2190 for the TPC-H queries

# DuckDB Benchmarks

DuckDB integration code is in `Benchmarks/DuckDB.hpp`.
The queries are in `Benchmarks/Benchmarks.cpp`, `duckdbQueries()`:
* lines 2404-2520 for the TPC-H queries

# Order-preservation indexes benchmark

This benchmark code can be found in `Benchmarks/MicroBenchmarks.cpp`.
