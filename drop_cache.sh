free -h
sync
echo 4 | tee /proc/sys/vm/drop_caches
echo 3 | tee /proc/sys/vm/drop_caches
free -h
sleep 1
