sudo sync
sudo sysctl -w vm.drop_caches=1
sudo sync
sudo sysctl -w vm.drop_caches=2
sudo sync
sudo sysctl -w vm.drop_caches=3

