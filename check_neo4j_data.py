#!/usr/bin/env python3
"""
Quick script to check Neo4j cloud database contents
"""

from neo4j import GraphDatabase
import os

# Your Neo4j Aura connection details
URI = "neo4j+s://98b288d4.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "HGww_ge5AFcjas-ak9gm47QFmt3a1U0A--z5sSqr0AU"

def check_database():
    """Check what data exists in the Neo4j database"""
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    
    try:
        with driver.session() as session:
            print("🔗 Connected to Neo4j Aura successfully!")
            print("=" * 50)
            
            # Check total nodes
            result = session.run("MATCH (n) RETURN count(n) as total_nodes")
            total_nodes = result.single()["total_nodes"]
            print(f"📊 Total nodes: {total_nodes}")
            
            # Check node types
            result = session.run("MATCH (n) RETURN DISTINCT labels(n) as node_types, count(n) as count ORDER BY count DESC")
            print("\n📋 Node types:")
            for record in result:
                labels = record["node_types"]
                count = record["count"]
                print(f"  {labels}: {count}")
            
            # Check relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC")
            print("\n🔗 Relationship types:")
            relationships = list(result)
            if relationships:
                for record in relationships:
                    print(f"  {record['rel_type']}: {record['count']}")
            else:
                print("  No relationships found")
            
            # Check companies if any exist
            result = session.run("MATCH (c:Company) RETURN c.symbol, c.name, c.sector LIMIT 10")
            companies = list(result)
            if companies:
                print("\n🏢 Sample companies:")
                for record in companies:
                    symbol = record["c.symbol"]
                    name = record["c.name"] 
                    sector = record["c.sector"]
                    print(f"  {symbol}: {name} ({sector})")
            else:
                print("\n🏢 No companies found in database")
            
            # Check schema
            print("\n🏗️ Database schema:")
            result = session.run("CALL db.schema.nodeTypeProperties()")
            schema_info = list(result)
            if schema_info:
                for record in schema_info:
                    print(f"  {record}")
            else:
                print("  No schema information available")
                
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    check_database()