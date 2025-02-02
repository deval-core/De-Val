#!/bin/bash

iptables -A OUTPUT -j DROP
iptables -I OUTPUT -o lo -j ACCEPT

iptables -I OUTPUT -d 172.17.0.1 -j ACCEPT
iptables -I OUTPUT -d 172.18.0.2 -j ACCEPT

GATEWAY_IP=$(ip route | grep default | awk '{print $3}')
iptables -I OUTPUT -d $GATEWAY_IP -j ACCEPT

iptables -I OUTPUT -p tcp -d huggingface.co -m tcp --dport 443 -j ACCEPT
iptables -I OUTPUT -p tcp -d cdn-lfs-us-1.hf.co -m tcp --dport 443 -j ACCEPT
iptables -I OUTPUT -p tcp -d cdn-lfs-eu-1.hf.co -m tcp --dport 443 -j ACCEPT
iptables -I OUTPUT -p tcp -d cdn-lfs.hf.co -m tcp --dport 443 -j ACCEPT

exec "$@"