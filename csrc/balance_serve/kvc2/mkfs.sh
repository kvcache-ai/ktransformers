sudo umount /mnt/xwy 
sudo mkfs.xfs /dev/nvme0n1 -f
sudo mount /dev/nvme0n1 /mnt/xwy
sudo chown -R xwy /mnt/xwy/