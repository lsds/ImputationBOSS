#~/bin/sh

index=(		0		1		2		3		4		5		6		7 			8		)
MB=(		0.1		1		10		100		1000	2000	5000	10000		100000	)
scale=(		0.0001	0.001	0.01	0.1		1.0		2.0		5.0		10.0		100.0	)

if ! [ -x "$(command -v python3)" ]; then
   echo 'Error: python3 is missing.' >&2
   exit 1
fi

for i in $(seq ${1-0} ${2-8}) ; do
   echo "generate SF ${scale[$i]}..."
   cd data/tpch_${MB[$i]}MB
   python3 ../dirtyLineitem.py
   cd ../..
done
