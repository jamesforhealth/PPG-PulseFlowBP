import sqlite3

def get_segment_info(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    SELECT ds.id, pis.identifier, ds.array_index
    FROM data_segment ds
    JOIN patient_info_snapshot pis ON ds.patient_snapshot_id = pis.id
    WHERE ds.data_source = (SELECT id FROM data_source WHERE name = 'PulseDB - Vital')
    ORDER BY pis.identifier, ds.array_index
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    conn.close()
    
    return results
def check_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    conn.close()
    
    print("Tables in the database:")
    for table in tables:
        print(table[0])

# 使用方法
db_path = 'PulseDB analysis test3.sqlite3'

segment_info = get_segment_info(db_path)

for segment_id, mat_file, array_index in segment_info:
    print(f"Segment ID: {segment_id}, MAT file: {mat_file}, Array Index: {array_index}")