#!/usr/bin/env python3
"""
Simple Neo4j connection test with various URI formats
"""

from neo4j import GraphDatabase
import socket

# Test configurations
configs = [
    {
        "name": "Neo4j Aura with neo4j+s",
        "uri": "neo4j+s://98b288d4.databases.neo4j.io",
        "user": "neo4j",
        "password": "wENxe9nIG0Kl8ysrxijtMEUTOoMD1HwrHqua5iOUk0o"
    },
    {
        "name": "Neo4j Aura with bolt+s",
        "uri": "bolt+s://98b288d4.databases.neo4j.io:7687",
        "user": "neo4j",
        "password": "wENxe9nIG0Kl8ysrxijtMEUTOoMD1HwrHqua5iOUk0o"
    },
    {
        "name": "Neo4j Aura without port",
        "uri": "bolt+s://98b288d4.databases.neo4j.io",
        "user": "neo4j",
        "password": "wENxe9nIG0Kl8ysrxijtMEUTOoMD1HwrHqua5iOUk0o"
    }
]

print("Testing Neo4j connection with different URI formats...")
print("=" * 60)

# First check DNS resolution
try:
    ip = socket.gethostbyname("98b288d4.databases.neo4j.io")
    print(f"✅ DNS Resolution: 98b288d4.databases.neo4j.io -> {ip}")
except Exception as e:
    print(f"❌ DNS Resolution failed: {e}")

print()

# Test each configuration
for config in configs:
    print(f"Testing: {config['name']}")
    print(f"URI: {config['uri']}")
    
    try:
        driver = GraphDatabase.driver(
            config['uri'],
            auth=(config['user'], config['password']),
            max_connection_lifetime=60,
            connection_timeout=10
        )
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record['test'] == 1:
                print(f"✅ Connection successful!")
                
                # Get some info
                result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                count = result.single()['count']
                print(f"   Total nodes: {count}")
                break
        
        driver.close()
        
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}")
    
    print()

print("=" * 60)