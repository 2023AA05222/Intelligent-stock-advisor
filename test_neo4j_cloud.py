#!/usr/bin/env python3
"""
Test Neo4j cloud connection
"""

from neo4j import GraphDatabase
import os

# Neo4j cloud credentials
NEO4J_URI = "neo4j+s://98b288d4.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "wENxe9nIG0Kl8ysrxijtMEUTOoMD1HwrHqua5iOUk0o"  # Correct cloud password

def test_connection():
    """Test Neo4j cloud connection"""
    print(f"Testing Neo4j connection...")
    print(f"URI: {NEO4J_URI}")
    print(f"Username: {NEO4J_USERNAME}")
    
    try:
        # Create driver
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' as message")
            record = result.single()
            print(f"✅ {record['message']}")
            
            # Get database info
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            for record in result:
                print(f"Database: {record['name']} {record['versions'][0]} ({record['edition']})")
            
            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()['count']
            print(f"Total nodes in database: {count}")
        
        driver.close()
        print("\n✅ Neo4j cloud connection is working!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()