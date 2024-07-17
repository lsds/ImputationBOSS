(Sort
 (Project
  (Group
   (Project
    (Project
     (Select
	  'LINEITEM
      (Where
	   (Greater
	    (DateObject "1998-08-31")
		'L_SHIPDATE)
	   )
	  )
      (As
	   'RETURNFLAG_AND_LINESTATUS
	   (StringJoin
	    'L_RETURNFLAG
		'L_LINESTATUS)
	   'L_QUANTITY
	   'L_QUANTITY
	   'L_EXTENDEDPRICE
	   'L_EXTENDEDPRICE
	   'L_DISCOUNT
	   'L_DISCOUNT
	   'calc1
	   (Minus
	    1.0
		'L_DISCOUNT)
	   'calc2
	   (Plus
	    1.0
		'L_TAX)
	   )
	 )
     (As
	  'RETURNFLAG_AND_LINESTATUS
	  'RETURNFLAG_AND_LINESTATUS
	  'L_QUANTITY
	  'L_QUANTITY
	  'L_EXTENDEDPRICE
	  'L_EXTENDEDPRICE
	  'L_DISCOUNT
	  'L_DISCOUNT
	  'DISC_PRICE
	  (Times
	   'L_EXTENDEDPRICE
	   'calc1)
	  'calc
	  'calc2
	 )
	)
    (By
	'RETURNFLAG_AND_LINESTATUS)
	(Sum 'L_QUANTITY)
	(Sum 'L_EXTENDEDPRICE)
	(Sum 'DISC_PRICE)
	(Sum
	 (Times
	  'DISC_PRICE
	  'calc)
	 )
	(Sum L_DISCOUNT)
	(Count 'L_QUANTITY)
   )
   (As
    'RETURNFLAG_AND_LINESTATUS
	'RETURNFLAG_AND_LINESTATUS
	'L_QUANTITY
	'L_QUANTITY
	'L_EXTENDEDPRICE
	'L_EXTENDEDPRICE
	'DISC_PRICE
	'DISC_PRICE
	'TimesDISC_PRICEcalc
	'TimesDISC_PRICEcalc
	'AVG_QTY
	(Divide 'L_QUANTITY 'CountL_QUANTITY)
	'AVG_PRICE
	(Divide 'L_EXTENDEDPRICE 'CountL_QUANTITY)
	'AVG_DISC
	(Divide 'DISC_PRICE 'CountL_QUANTITY)
	'CountL_QUANTITY
	'CountL_QUANTITY
   )
  )
 (By 'RETURNFLAG_AND_LINESTATUS)
)
