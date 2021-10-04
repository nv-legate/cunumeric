rm -rf prof*
rm -rf spy*
rm -rf legion_prof*
#prof
../../../legate.core/install/bin/legate --cpus 1 test.py -lg:prof 1 -lg:prof_logfile prof_%.gz

#spy
../../../legate.core/install/bin/legate --cpus 1 test.py -lg:spy -logfile spy_%.log
../../../legate.core/legion/tools/legion_spy.py  -dez spy_*.log

#both
../../../legate.core/legion/tools/legion_prof.py prof_*.gz spy_*.log
cp dataflow_legion_python* legion_prof/
cp event_graph_legion_python* legion_prof/
echo "done"
