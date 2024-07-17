#lang racket
(module* main cli
  (require racket/port
           racket/pretty
           racket/list
           racket/match
           racket/contract
           gregor
           (only-in srfi/19 string->date)
           (only-in seq nth)
           racket/date
           racket/string
           racket/function)
  (flag (expression-percentage #:param [expression-percentage 10] n)
        ("-e" "--expressions" "percentage of values in table that are wrapped in expressions")
        (expression-percentage (string->number n)))
  (define (eval-required) (> (expression-percentage) 0))
  (program
   (run-query
    [data-path "Path to tpc-h data"]
    [query-file " The path to the query file"])
   (define load-data #t)
   (define schema
     (hash 'LINEITEM (list (list 'L_ORDERKEY 'INTEGER)
                           (list 'L_PARTKEY     'INTEGER)
                           (list 'L_SUPPKEY     'INTEGER)
                           (list 'L_LINENUMBER  'INTEGER)
                           (list 'L_QUANTITY    'DECIMAL)
                           (list 'L_EXTENDEDPRICE  'DECIMAL)
                           (list 'L_DISCOUNT    'DECIMAL)
                           (list 'L_TAX         'DECIMAL)
                           (list 'L_RETURNFLAG  'CHAR)
                           (list 'L_LINESTATUS  'CHAR)
                           (list 'L_SHIPDATE    'DATE)
                           (list 'L_COMMITDATE  'DATE)
                           (list 'L_RECEIPTDATE 'DATE)
                           (list 'L_SHIPINSTRUCT 'CHAR)
                           (list 'L_SHIPMODE     'CHAR)
                           (list 'L_COMMENT      'VARCHAR)
                           )
           'NATION (list
                    (list 'N_NATIONKEY 'INTEGER)
                    (list 'N_NAME 'CHAR)
                    (list 'N_REGIONKEY 'INTEGER)
                    (list 'N_COMMENT 'VARCHAR)
                    )


           'SUPPLIER (list
                      (list 'S_SUPPKEY 'INTEGER)
                      (list 'S_NAME 'CHAR)
                      (list 'S_ADDRESS 'VARCHAR)
                      (list 'S_NATIONKEY 'INTEGER)
                      (list 'S_PHONE 'CHAR)
                      (list 'S_ACCTBAL 'DECIMAL)
                      (list 'S_COMMENT 'VARCHAR)
                      )
           'PART (list

                  (list 'P_PARTKEY 'INTEGER)
                  (list 'P_NAME 'VARCHAR)
                  (list 'P_MFGR 'CHAR)
                  (list 'P_BRAND 'CHAR)
                  (list 'P_TYPE 'VARCHAR)
                  (list 'P_SIZE 'INTEGER)
                  (list 'P_CONTAINER 'CHAR)
                  (list 'P_RETAILPRICE 'DECIMAL)
                  (list 'P_COMMENT 'VARCHAR)

                  )
           'PARTSUPP (list
                      (list 'PS_PARTKEY 'INTEGER)
                      (list 'PS_SUPPKEY 'INTEGER)
                      (list 'PS_AVAILQTY 'INTEGER)
                      (list 'PS_SUPPLYCOST 'DECIMAL)
                      (list 'PS_COMMENT 'VARCHAR)

                      )
           'ORDERS (list
                    (list 'O_ORDERKEY 'INTEGER)
                    (list 'O_CUSTKEY 'INTEGER)
                    (list 'O_ORDERSTATUS 'CHAR)
                    (list 'O_TOTALPRICE 'DECIMAL)
                    (list 'O_ORDERDATE 'DATE)
                    (list 'O_ORDERPRIORITY 'CHAR)
                    (list 'O_CLERK 'CHAR)
                    (list 'O_SHIPPRIORITY 'INTEGER)
                    (list 'O_COMMENT 'VARCHAR)

                    )

           'CUSTOMER (list
                      (list 'C_CUSTKEY 'INTEGER)
                      (list 'C_NAME 'VARCHAR)
                      (list 'C_ADDRESS 'VARCHAR)
                      (list 'C_NATIONKEY 'INTEGER)
                      (list 'C_PHONE 'CHAR)
                      (list 'C_ACCTBAL 'DECIMAL)
                      (list 'C_MKTSEGMENT 'CHAR)
                      (list 'C_COMMENT 'VARCHAR)))
     )
   (define parse-functions (hash
                            'DECIMAL 'string->number
                            'CHAR 'identity
                            'VARCHAR 'identity
                            'DATE 'iso8601->date
                            'INTEGER 'string->number
                            ))


   (define (get-memoized-default-projection table)
     (for/list ([i (in-range +inf.0)] [attribute (map car (hash-ref schema table))])
       (list attribute `(hash-ref (hash-ref data (quote ,table)) (quote ,attribute)))))

   (define (get-default-projection table)
     (for/list ([i (in-range +inf.0)] [attribute (map car (hash-ref schema table))])
       (list attribute `(,(hash-ref parse-functions
                                    (cadr (assoc attribute (hash-ref schema table)))) (nth ,i line)))))

   (define (get-projected-pattributes table)
     (map car (hash-ref schema table))
     )

   (define (type-of-scalar expression)
     (match expression
       [`(DateObject ,_) 'DATE]
       [`(Date ,_) 'DATE]
       [(? symbol? s) (for/first ([(table attributes) schema] #:when (hash-has-key? schema s)) (hash-ref s))]
       [_ 'UNKNOWN]
       )
     )

   (define (rewrite-parameter expression [eval-symbols #f])
     (match expression
       [`(DateObject ,dateString) #`(iso8601->date #,dateString)]
       [`(Date ,dateString) #`(iso8601->date #,dateString)]
       [`(Times ,arguments ...) #`(* #,@(map (curryr rewrite-parameter eval-symbols) arguments))]
       [`(Plus ,arguments ...) #`(+ #,@(map (curryr rewrite-parameter eval-symbols) arguments))]
       [`(Subtract ,arguments ...) #`(- #,@(map (curryr rewrite-parameter eval-symbols) arguments))]
       [`(Multiply ,arguments ...) #`(* #,@(map (curryr rewrite-parameter eval-symbols) arguments))]
       [`(Divide ,arguments ...) #`(/ #,@(map (curryr rewrite-parameter eval-symbols) arguments))]
       [`(Year ,argument) #`(->year #,((curryr rewrite-parameter eval-symbols) argument))]
       [`(Minus ,arguments ...) #`(- #,@(map (curryr rewrite-parameter eval-symbols) arguments))]
       [`(StringJoin ,arguments ...) #`(string-append #,@(map (curryr rewrite-parameter eval-symbols) arguments) )]
       [`(As ,left ,right ,remaining ...)
        #`([#,((curryr rewrite-parameter #f) left ) #,((curryr rewrite-parameter eval-symbols) right )]
           #,@((curryr rewrite-parameter eval-symbols) `(As ,@remaining) ))]
       [`(As) #`()]
       [`(Greater ,left ,right)
        #`(#,(if (or (eq? (type-of-scalar left) 'DATE) (eq? (type-of-scalar right) 'DATE)) #'date>? #'>)
           #,((curryr rewrite-parameter eval-symbols) left )
           #,((curryr rewrite-parameter eval-symbols) right ))]
       [`(Equal ,left ,right)
        #`(#,(if (or (eq? (type-of-scalar left) 'DATE) (eq? (type-of-scalar right) 'DATE))
                 #'date=? #'eq?) #,((curryr rewrite-parameter eval-symbols) left ) #,((curryr rewrite-parameter eval-symbols) right ))]
       [`(StringContains ,the-string ,the-substring)
        #`(string-contains? #,((curryr rewrite-parameter eval-symbols) the-string ) #,((curryr rewrite-parameter eval-symbols) the-substring ))]
       [`(And ,predicates ...) #`(and #,@(map (lambda (predicate) ((curryr rewrite-parameter eval-symbols) predicate )) predicates))]
       [`(Function ,arg ,body) ((curryr rewrite-parameter eval-symbols) body)]
       [`(Column ,tuple ,position) #`(nth #,(sub1 position) #,tuple)]
       [`(List ,values ...) #`(list #,@(map (curryr rewrite-parameter eval-symbols) values))]
       [`(quote ,(? symbol? the-symbol)) (if eval-symbols #`(eval #,the-symbol) the-symbol)]
       [(? symbol? the-symbol) (if eval-symbols #`(eval  #,expression) expression)]
       [_ expression]))


   (define (get-aggregate-symbol aggregate)
     (string->symbol (string-join (map symbol->string
                                       (remove*
                                        (list 'quote 'Sum)
                                        (flatten aggregate))) "")))

   (define (initialize-aggregation-tables aggregates)
     (for/list
                 ([aggregate aggregates])
       `(,(string->symbol (string-join (map symbol->string
                                            (flatten aggregate)) "" #:after-last "Hash"))
         (make-hash))))

   (define (get-aggregation-table-values aggregates group-by-attributes)
     (for/list
                 ([aggregate aggregates])
       `(,(get-aggregate-symbol aggregate)
         (hash-ref
          ,(string->symbol (string-join
                            (map symbol->string (flatten aggregate))
                            "" #:after-last "Hash")) ,@group-by-attributes))))

   (define (get-aggregation-table-keys aggregate group-by-attribute)
     `(,@group-by-attribute
       (hash-keys
        ,(string->symbol
          (string-join
           (map symbol->string (flatten aggregate)) "" #:after-last "Hash")))))

   (define (turn-into-result-tuple-let-statement attributes)
     `(result (list ,@(for/list ([i (range 1 (length attributes) 2)]) (nth i attributes))))
     )

   (define (get-tuple-representation-from-attributes attributes)
     (for/list ([i (range 1 (length attributes) 2)]) (nth i attributes))
     )

   (define (rewrite-aggregations aggregates grouping-attribute)
     (let ([accumulators #hash((Sum . +) (Count . (lambda (old agg) (add1 old))))])
       (for/list
                 ([aggregate aggregates])
         `(hash-update!
           ,(string->symbol (string-join (map symbol->string
                                              (flatten aggregate)) "" #:after-last "Hash"))
           (list ,@ grouping-attribute)
           (lambda (old) (,(hash-ref accumulators (first aggregate)) old ,(rewrite-parameter (second aggregate) #t)))
           (const 0)
           )))
     )

   (define/contract (rewrite-attributes input)
     ((or/c list? symbol?) . -> . (or/c list? symbol?))
     (match input
       [`(quote ,(? symbol? the-symbol)) the-symbol]
       [(list attributes ...)
        (for/list ([attribute attributes])
          (if (and (list? attribute) (eq? (length attribute) 2) (eq? (first attribute) 'quote) (symbol? (second attribute))) (second attribute) attribute)
          )]
       [_ input]
       )

     )

   (define/contract (rewrite-operator expression pushTarget)
     (list? procedure? . -> . syntax?)
     (match expression
       [`(Sort ,input ,by) (rewrite-operator input pushTarget)]
       [`(Join ,left ,right (Where (Equal ,firstAttribute ,secondAttribute)))
        (let ([hashtable-symbol (gensym (format "joinTable~s~s" (rewrite-attributes firstAttribute) (rewrite-attributes secondAttribute)))]
              [left-attributes (list)]
              [build-relation-tuple (gensym)])
          #`((let ([#,hashtable-symbol (make-hash)])
               #,@(rewrite-operator
                   left
                   (lambda (tuple-representation-attributes-befor-deduplication)
                     (let ([tuple-representation-attributes
                            (remove-duplicates tuple-representation-attributes-befor-deduplication)])
                       (set! left-attributes tuple-representation-attributes)
                       #`((hash-update!
                           #,hashtable-symbol
                           #,(rewrite-parameter firstAttribute #t)
                           (lambda (old-value)
                             (list* (list #,@tuple-representation-attributes) old-value))
                           (const (list)))
                          )))
                   )

               #,@(rewrite-operator
                   right
                   (lambda (tuple-representation-attributes)
                     #`(
                        (for
                                    ([#,build-relation-tuple
                                      (hash-ref #,hashtable-symbol
                                                #,(rewrite-parameter secondAttribute #t) (list))])
                          (let #,(for/list ([i (in-range +inf.0)]
                                            [attribute left-attributes])
                                   `[,attribute (nth ,i ,build-relation-tuple)])
                              #,@(pushTarget
                                  (append left-attributes tuple-representation-attributes))) )
                        )))

               ))
          )]
       [`(Order ,input ,by) (rewrite-operator input pushTarget)]
       [`(Top ,input (By ,by ...) ,n)
        (let ([iterator (gensym)]
              [attributes (list)]
              [heap (gensym)])
          #`((let ([#,heap
                    (make-heap (lambda (left right)
                                 (or
                                  (> (nth 0 left) (nth 0 right))
                                  #,@(for/list ([i (range 1 (length by))])
                                       #`(and
                                          (= (nth #,(sub1 i) left)
                                             (nth #,(sub1 i) right)
                                             ) (> (nth #,i left) (nth #,i right)))))
                                 ))])
               #,@(rewrite-operator
                   input
                   (lambda (before-attributes)
                     (set! attributes before-attributes)
                     #`((heap-add! #,heap
                                   (list #,@(map (curryr rewrite-parameter #t) by) #,@before-attributes))
                        (if (> (heap-count #,heap) #,n) (heap-remove-min! #,heap) '()))
                     ))
               (for ([#,iterator (in-heap/consume! #,heap)])
                 (let (#,@(for/list ([i (in-range +inf.0)]
                                     [attribute (append* by (list attributes))]
                                     #:when (>= i (length by)))
                            #`[#,attribute (nth #,i #,iterator)]
                            ))
                   #,@(pushTarget attributes))
                 )

               )))]
       [`(Group ,input (By ,attributes ...) ,aggregates ...)
        #`((let #,(initialize-aggregation-tables aggregates)
               #,@(rewrite-operator
                   input
                   (lambda (tuple-representation-attributes)
                     (rewrite-aggregations aggregates (rewrite-attributes attributes))))
             #,(let ([key-symbol (gensym)])
                 #`(for (#,(get-aggregation-table-keys (first aggregates)
                                                       (list key-symbol)))
                     (let* (#,@(get-aggregation-table-values aggregates (list key-symbol))
                            #,@(for/list ([i (in-range +inf.0)]
                                          [attribute attributes])
                                 #`(#,(rewrite-attributes attribute)
                                    (nth  #,i #,key-symbol)))
                            )
                       #,@(pushTarget (append (rewrite-attributes attributes)
                                              (map get-aggregate-symbol aggregates)
                                              )))))))]
       [`(Group ,input ,aggregate)
        #`(( let ([result 0])
                 #,@(rewrite-operator
                     input
                     (const
                      #`((set! result
                               (+ result #,(rewrite-parameter (last aggregate)
                                                              (eval-required)))))))
                 #,@(pushTarget (list 'result))))]
       [`(Project ,input ,attributes)
        (rewrite-operator
         input
         (lambda (tuple-representation-attributes)
           #`((let* ([tuple (list #,@tuple-representation-attributes)]
                     #,@(rewrite-parameter attributes (eval-required))
                     )
                #,@(pushTarget
                    (get-tuple-representation-from-attributes (last attributes)))))))]
       [`(Select ,input ,predicate)
        (rewrite-operator
         input
         (lambda (tuple-representation-attributes)
           #`((if
               #,(rewrite-parameter
                  (last predicate)
                  (eval-required))
               #,@(pushTarget tuple-representation-attributes)  (list)))))]
       [`(quote ,(? symbol? input-symbol)) (rewrite-operator input-symbol pushTarget)]
       [(? symbol? input-symbol)
        #`((begin
             #,@(if load-data
                    (list
                     #`(if
                        (not (hash-has-key? data (quote #,input-symbol)))
                        (csv-for-each
                         (lambda (line)
                           (let #,(get-default-projection input-symbol)
                               #,@(map
                                   (lambda (attribute)
                                     `(hash-update!
                                       (hash-ref! data (quote ,input-symbol) (make-hash))
                                       (quote ,(first attribute))
                                       (lambda (old-value)
                                         (list*
                                          (if (< (random 100) ,(expression-percentage))
                                              (quasiquote
                                               (identity (unquote ,(first attribute))))
                                              ,(first attribute)) old-value))
                                       (const (list))))
                                   (hash-ref schema input-symbol))
                             ))
                         (make-csv-reader
                          (open-input-file
                           #,(format "~a/~a.tbl" data-path
                                     (string-downcase (symbol->string input-symbol))))
                          '((separator-chars #\|) ))
                         )
                        '())) '())
             (for #,(get-memoized-default-projection input-symbol)
                  #,@(pushTarget (get-projected-pattributes input-symbol)))))]
       [_ expression])
     )
   (define ns (make-base-namespace))
   (eval '(require racket/function (only-in srfi/19 string->date)
                   (for-syntax racket/base) csv-reading racket/date seq
                   gregor
                   data/heap
                   racket/string racket/block) ns)
   (eval '(define data (make-hash)) ns)
   (let* ((plan (read (open-input-file query-file)))
          (s (rewrite-operator plan (lambda (tuple-attributes)
                                      #`((displayln (list #,@tuple-attributes))))))
          (v (pretty-print  (syntax->datum s)))
          (query (car (syntax->datum s))))
     (time (eval query ns))
     (for ((i (in-range 15))) (time (eval query ns)))

     ))
  (run run-query))
