#!/bin/bash

# Allow loopback traffic
iptables -A OUTPUT -o lo -j ACCEPT

# Allow traffic to the Hugging Face endpoint (replace with actual IP/domain)
iptables -A OUTPUT -p tcp -d https://huggingface.co --dport 443 -j ACCEPT

# Allow traffic to the local API endpoint
iptables -A OUTPUT -p tcp -d 0.0.0.0 --dport 8000 -j ACCEPT

# Block all other outgoing traffic
iptables -A OUTPUT -j DROP

# Execute the original command
exec "$@"