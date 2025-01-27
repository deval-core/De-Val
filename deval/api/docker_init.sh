#!/bin/bash

# Get the Hugging Face endpoint's IP address
HUGGINGFACE_IP=$(getent ahosts https://huggingface.co | awk '{print $1}' | head -n 1)

# Allow local communication (e.g., 0.0.0.0:8000)
iptables -A OUTPUT -p tcp --dport 8000 -j ACCEPT

# Allow traffic to Hugging Face endpoint
iptables -A OUTPUT -d $HUGGINGFACE_IP -j ACCEPT

# Drop all other outbound traffic
iptables -P OUTPUT DROP

# Optional: Log dropped packets (for debugging purposes)
iptables -A OUTPUT -j LOG --log-prefix "Blocked Outbound: " --log-level 4

# Execute the main process
exec "$@"
