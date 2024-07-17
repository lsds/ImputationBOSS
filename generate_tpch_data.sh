#~/bin/sh

index=(		0		1		2		3		4		5		6		7 			8		)
MB=(		0.1		1		10		100		1000	2000	5000	10000		100000	)
scale=(		0.0001	0.001	0.01	0.1		1.0		2.0		5.0		10.0		100.0	)

echo "compile dbgen..."
cd dbgen
make
if [[ $? -ne 0 ]]; then
   echo "check the compilation options in dbgen/Makefile and try again." >&2
      exit 1
fi

#if ! [ -x "$(command -v csvformat)" ]; then
#   echo "install csvkit..."
#   if ! [ -x "$(command -v pip)" ]; then
#      echo 'Error: pip is missing. install pip or manually install csvkit.' >&2
#      exit 1
#   fi
#   sudo pip install csvkit
#fi
#if ! [ -x "$(command -v csvformat)" ]; then
#   echo 'Error: csvformat command is missing.' >&2
#   exit 1
#fi

rm -rf *.tbl
for i in $(seq ${1-0} ${2-8}); do
   echo "[$i / ${2-8}] generate TPC-H SF ${scale[$i]}..."
   mkdir -p ../data/tpch_${MB[$i]}MB
   ./dbgen -s ${scale[$i]}
   #./dbgen -O h -s ${scale[$i]}
   #echo "convert tbl file to csv file..."
   #for file in `ls *.tbl`; do
   #   csvformat -d '|' -D ',' $file | sed 's/,$//' > ../data/tpch_${MB[$i]}MB/${file/tbl/csv}
   #   echo "'${file/tbl/csv}' created."
   #   rm $file
   #done
   mv -f *.tbl ../data/tpch_${MB[$i]}MB/
done
