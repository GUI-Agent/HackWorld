#!/bin/bash
# List all running VMs
vms=$(vmrun list | tail -n +2)

# Loop and stop each VM
for vm in $vms; do
    echo "Stopping VM: $vm"
    vmrun stop "$vm" soft
done
