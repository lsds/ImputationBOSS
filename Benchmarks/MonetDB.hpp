#pragma once

#include <embedded/embedded.h>

#include <filesystem>
#include <string>

#ifndef _WIN32
#define LLFMT "%lld"
#else
#define LLFMT "%I64d"
#endif

static void printResult(monetdb_result* result, unsigned int maxRows = (unsigned int)(-1)) {
  for(size_t r = 0; r < result->nrows && r < maxRows; r++) {
    for(size_t c = 0; c < result->ncols; c++) {
      monetdb_column* actual_column = monetdb_result_fetch(result, c);
      switch(actual_column->type) {
      case monetdb_int8_t: {
        monetdb_column_int8_t* col = (monetdb_column_int8_t*)actual_column;
        printf("%d", (int)col->data[r]);
        break;
      }
      case monetdb_int16_t: {
        monetdb_column_int16_t* col = (monetdb_column_int16_t*)actual_column;
        printf("%d", (int)col->data[r]);
        break;
      }
      case monetdb_int32_t: {
        monetdb_column_int32_t* col = (monetdb_column_int32_t*)actual_column;
        printf("%d", (int)col->data[r]);
        break;
      }
      case monetdb_int64_t: {
        monetdb_column_int64_t* col = (monetdb_column_int64_t*)actual_column;
        printf(LLFMT, (long long int)col->data[r]);
        break;
      }
      case monetdb_float: {
        monetdb_column_float* col = (monetdb_column_float*)actual_column;
        printf("%f", col->data[r]);
        break;
      }
      case monetdb_double: {
        monetdb_column_double* col = (monetdb_column_double*)actual_column;
        printf("%lf", col->data[r]);
        break;
      }
      case monetdb_str: {
        monetdb_column_str* col = (monetdb_column_str*)actual_column;
        printf("%s", col->data[r] ? col->data[r] : "NULL");
        break;
      }
      default: {
        printf("UNKNOWN");
      }
      }

      if(c + 1 < result->ncols) {
        printf(", ");
      }
    }
    printf("\n");
  }
}

class MonetDBHandling {
public:
  MonetDBHandling(int size, bool enableIndexes, bool enableMultithreading,
                  bool useFixedPoint = false) {
    // make sure it is a clean folder
    cleanupTempFolder();
    auto absFarmPath = std::filesystem::absolute("monetdbfarm").string();
    auto error = monetdb_startup(absFarmPath.data(), 0, 0);
    if(error != 0) {
      throw std::runtime_error("MonetDB Init failed: " + std::string(error));
    }

    connection = monetdb_connect();

    monetdb_result* result = nullptr;

#ifndef NDEBUG
    std::string selectOptimizer = "select optimizer;";
    error = monetdb_query(connection, selectOptimizer.data(), 1, &result, NULL, NULL);
    if(error != 0) {
      throw std::runtime_error("SELECT optimizer failed: " + std::string(error));
    }
    printResult(result, 1);
#endif // !NDEBUG

    if(!enableMultithreading) {
      std::string setOptimizer = "set optimizer='minimal_pipe'";
      error = monetdb_query(connection, setOptimizer.data(), 1, NULL, NULL, NULL);
      if(error != 0) {
        throw std::runtime_error("MonetDB SET optimizer failed: " + std::string(error));
      }
#ifndef NDEBUG
      else {
        std::cout << "optimizer set to 'sequential_pipe'" << std::endl;
      }
#endif // !NDEBUG
    }

    std::string numeric_type = useFixedPoint ? "DECIMAL(15,2)" : "DOUBLE";

    result = nullptr;
    std::string startTransaction = "START TRANSACTION;";
    error = monetdb_query(connection, startTransaction.data(), 1, NULL, NULL, NULL);
    if(error != 0) {
      throw std::runtime_error("MonetDB start transaction failed: " + std::string(error));
    }
    std::vector<std::string> createCmds{
        " CREATE TABLE region  ( r_regionkey  INTEGER NOT NULL,"
        "                             r_name       CHAR(25) NOT NULL,"
        "                             r_comment    VARCHAR(152));",
        " CREATE TABLE nation  ( n_nationkey  INTEGER NOT NULL,"
        "                             n_name       CHAR(25) NOT NULL,"
        "                             n_regionkey  INTEGER NOT NULL,"
        "                             n_comment    VARCHAR(152));",
        " CREATE TABLE part  ( p_partkey     BIGINT NOT NULL,"
        "                           p_name        VARCHAR(55) NOT NULL,"
        "                           p_mfgr        CHAR(25) NOT NULL,"
        "                           p_brand       CHAR(10) NOT NULL,"
        "                           p_type        VARCHAR(25) NOT NULL,"
        "                           p_size        INTEGER NOT NULL,"
        "                           p_container   CHAR(10) NOT NULL,"
        "                           p_retailprice " +
            numeric_type +
            " NOT NULL,"
            "                           p_comment     VARCHAR(23) NOT NULL);",
        " CREATE TABLE supplier ( s_suppkey     BIGINT NOT NULL,"
        "                              s_name        CHAR(25) NOT NULL,"
        "                              s_address     VARCHAR(40) NOT NULL,"
        "                              s_nationkey   INTEGER NOT NULL,"
        "                              s_phone       CHAR(15) NOT NULL,"
        "                              s_acctbal     " +
            numeric_type +
            " NOT NULL,"
            "                              s_comment     VARCHAR(101) NOT NULL);",
        " CREATE TABLE partsupp ( ps_partkey     BIGINT NOT NULL,"
        "                              ps_suppkey     BIGINT NOT NULL,"
        "                              ps_availqty    BIGINT NOT NULL,"
        "                              ps_supplycost  " +
            numeric_type +
            "  NOT NULL,"
            "                              ps_comment     VARCHAR(199) NOT NULL);",
        " CREATE TABLE customer ( c_custkey     BIGINT NOT NULL,"
        "                              c_name        VARCHAR(25) NOT NULL,"
        "                              c_address     VARCHAR(40) NOT NULL,"
        "                              c_nationkey   INTEGER NOT NULL,"
        "                              c_phone       CHAR(15) NOT NULL,"
        "                              c_acctbal     " +
            numeric_type +
            "   NOT NULL,"
            "                              c_mktsegment  CHAR(10) NOT NULL,"
            "                              c_comment     VARCHAR(117) NOT NULL);",
        " CREATE TABLE orders  ( o_orderkey       BIGINT NOT NULL,"
        "                            o_custkey        BIGINT NOT NULL,"
        "                            o_orderstatus    CHAR(1) NOT NULL,"
        "                            o_totalprice     " +
            numeric_type +
            " NOT NULL,"
            "                            o_orderdate      DATE NOT NULL,"
            "                            o_orderpriority  CHAR(15) NOT NULL,  "
            "                            o_clerk          CHAR(15) NOT NULL, "
            "                            o_shippriority   INTEGER NOT NULL,"
            "                            o_comment        VARCHAR(79) NOT NULL);",
        " CREATE TABLE lineitem ( l_orderkey    BIGINT NOT NULL,"
        "                              l_partkey     BIGINT NOT NULL,"
        "                              l_suppkey     BIGINT NOT NULL,"
        "                              l_linenumber  BIGINT NOT NULL,"
        "                              l_quantity    " +
            numeric_type +
            " NOT NULL,"
            "                              l_extendedprice  " +
            numeric_type +
            " NOT NULL,"
            "                              l_discount    " +
            numeric_type +
            " NOT NULL,"
            "                              l_tax         " +
            numeric_type +
            " NOT NULL,"
            "                              l_returnflag  CHAR(1) NOT NULL,"
            "                              l_linestatus  CHAR(1) NOT NULL,"
            "                              l_shipdate    DATE NOT NULL,"
            "                              l_commitdate  DATE NOT NULL,"
            "                              l_receiptdate DATE NOT NULL,"
            "                              l_shipinstruct CHAR(25) NOT NULL,"
            "                              l_shipmode     CHAR(10) NOT NULL,"
            "                              l_comment      VARCHAR(44) NOT NULL);"};
    for(auto& cmd : createCmds) {
      error = monetdb_query(connection, cmd.data(), 1, NULL, NULL, NULL);
      if(error != 0) {
        throw std::runtime_error("MonetDB create table failed: " + std::string(error));
      }
    }
    std::vector<std::string> pkCmds{"ALTER TABLE part"
                                    "  ADD CONSTRAINT part_kpey"
                                    "     PRIMARY KEY (p_partkey);",
                                    "ALTER TABLE supplier"
                                    "  ADD CONSTRAINT supplier_pkey"
                                    "     PRIMARY KEY (s_suppkey);",
                                    "ALTER TABLE partsupp"
                                    "  ADD CONSTRAINT partsupp_pkey"
                                    "     PRIMARY KEY (ps_partkey, ps_suppkey);",
                                    "ALTER TABLE customer"
                                    "  ADD CONSTRAINT customer_pkey"
                                    "     PRIMARY KEY (c_custkey);",
                                    "ALTER TABLE orders"
                                    "  ADD CONSTRAINT orders_pkey"
                                    "     PRIMARY KEY (o_orderkey);",
                                    "ALTER TABLE lineitem"
                                    "  ADD CONSTRAINT lineitem_pkey"
                                    "     PRIMARY KEY (l_orderkey, l_linenumber);",
                                    "ALTER TABLE nation"
                                    "  ADD CONSTRAINT nation_pkey"
                                    "     PRIMARY KEY (n_nationkey);",
                                    "ALTER TABLE region"
                                    "  ADD CONSTRAINT region_pkey"
                                    "     PRIMARY KEY (r_regionkey);"};
    if(enableIndexes) {
      for(auto& cmd : pkCmds) {
        error = monetdb_query(connection, cmd.data(), 1, NULL, NULL, NULL);
        if(error != 0) {
          throw std::runtime_error("MonetDB add primary key error: " + std::string(error));
        }
      }
    }
    std::vector<std::string> fkCmds{
        "ALTER TABLE supplier"
        "  ADD CONSTRAINT supplier_nation_fkey"
        "   FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey);",
        "ALTER TABLE partsupp"
        "  ADD CONSTRAINT partsupp_part_fkey"
        "   FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey);",
        "ALTER TABLE partsupp"
        "  ADD CONSTRAINT partsupp_supplier_fkey"
        "   FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey);",
        "ALTER TABLE customer"
        "  ADD CONSTRAINT customer_nation_fkey"
        "   FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey);",
        "ALTER TABLE orders"
        "  ADD CONSTRAINT orders_customer_fkey"
        "   FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey);",
        "ALTER TABLE lineitem"
        "  ADD CONSTRAINT lineitem_orders_fkey"
        "   FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey);",
        "ALTER TABLE lineitem"
        "  ADD CONSTRAINT lineitem_partsupp_fkey"
        "   FOREIGN KEY (l_partkey,l_suppkey)"
        "    REFERENCES partsupp(ps_partkey,ps_suppkey);",
        "ALTER TABLE nation"
        "  ADD CONSTRAINT nation_region_fkey"
        "   FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey);",
    };
    if(enableIndexes) {
      for(auto& cmd : fkCmds) {
        error = monetdb_query(connection, cmd.data(), 1, NULL, NULL, NULL);
        if(error != 0) {
          throw std::runtime_error("MonetDB add foreign key error: " + std::string(error));
        }
      }
    }
    std::vector<std::string> tables{"region",   "nation",   "part",   "supplier",
                                    "partsupp", "customer", "orders", "lineitem"};
    for(auto& table : tables) {
      std::string path = "../data/tpch_" + std::to_string(size) + "MB/" + table + ".tbl";
      auto absFilepath = std::filesystem::absolute(path).string();
      auto loadingQuery =
          "COPY INTO " + table + " FROM '" + absFilepath + "' USING DELIMITERS '|', '\n';";
      error = monetdb_query(connection, loadingQuery.data(), 1, NULL, NULL, NULL);
      if(error != 0) {
        throw std::runtime_error("MonetDB file loading failed: " + std::string(error));
      }
    }
    std::string commit = "COMMIT;";
    error = monetdb_query(connection, commit.data(), 1, NULL, NULL, NULL);
    if(error != 0) {
      throw std::runtime_error("MonetDB commit failed: " + std::string(error));
    }
  }

  void cleanupTempFolder() {
#ifdef WIN32
    system("rd /s /q \"./monetdbfarm\"");
#else
    system("rm -rf ./monetdbfarm/*");
#endif
  }

  ~MonetDBHandling() {
    monetdb_disconnect(connection);
    monetdb_shutdown();
    cleanupTempFolder();
  }

  monetdb_connection& getConnection() { return connection; }

private:
  monetdb_connection connection;
};

static auto& MonetDBhandlingPtr() {
  static std::unique_ptr<MonetDBHandling> monetDBhandling;
  return monetDBhandling;
}

static void releaseMonetDB() { MonetDBhandlingPtr().reset(); }

static auto& initMonetDB(size_t size, bool enableIndexes, bool enableMultithreading,
                         bool useFixedPoint) {
  static auto lastSize = size;
  static auto lastEnableIndexes = enableIndexes;
  static auto lastEnableMultithreading = enableMultithreading;
  static auto lastUseFixedPoint = useFixedPoint;
  if(!MonetDBhandlingPtr() || size != lastSize || enableIndexes != lastEnableIndexes ||
     enableMultithreading != lastEnableMultithreading || useFixedPoint != lastUseFixedPoint) {
    lastSize = size;
    lastEnableIndexes = enableIndexes;
    lastEnableMultithreading = enableMultithreading;
    lastUseFixedPoint = useFixedPoint;
    MonetDBhandlingPtr().reset();
    MonetDBhandlingPtr() =
        std::make_unique<MonetDBHandling>(size, enableIndexes, enableMultithreading, useFixedPoint);
  }
  return MonetDBhandlingPtr()->getConnection();
}