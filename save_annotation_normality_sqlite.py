import sqlite3
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import numpy as np
import h5py
import os
from tqdm import tqdm
from dtaidistance import dtw
from abp_anomaly_detection_algorithms import calculate_dtw_scores
import multiprocessing
from functools import partial
import time
import logging
import queue


# 设置日志
# logging.basicConfig(filename='process_log.txt', level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# def create_database(conn):
#     # conn = sqlite3.connect('pulsedb_annotations.db')
#     # conn = sqlite3.connect('pulsedb_annotations_test.db')
#     c = conn.cursor()
    
#     c.execute('''CREATE TABLE IF NOT EXISTS data_sources
#                  (id INTEGER PRIMARY KEY,
#                   name TEXT UNIQUE)''')
    
#     c.execute("INSERT OR IGNORE INTO data_sources (name) VALUES ('PulseDB_Vital')")
#     c.execute("INSERT OR IGNORE INTO data_sources (name) VALUES ('PulseDB_MIMIC')")
    
#     c.execute('''CREATE TABLE IF NOT EXISTS mat_files
#                  (id INTEGER PRIMARY KEY, 
#                   data_source_id INTEGER,
#                   filename TEXT,
#                   total_segments INTEGER,
#                   FOREIGN KEY (data_source_id) REFERENCES data_sources(id),
#                   UNIQUE (data_source_id, filename))''')
    
#     c.execute('''CREATE TABLE IF NOT EXISTS segment_annotations
#                  (id INTEGER PRIMARY KEY,
#                   file_id INTEGER,
#                   segment_index INTEGER,
#                   status INTEGER,
#                   max_dtw_score REAL,
#                   FOREIGN KEY (file_id) REFERENCES mat_files(id))''')
    
#     c.execute("CREATE INDEX IF NOT EXISTS idx_file_segments ON segment_annotations(file_id, segment_index)")
    
#     conn.commit()
#     return conn

# def create_database(db_name):
#     conn = sqlite3.connect(db_name)
#     c = conn.cursor()
    
#     c.execute('''CREATE TABLE IF NOT EXISTS data_sources
#                  (id INTEGER PRIMARY KEY,
#                   name TEXT UNIQUE)''')
    
#     c.execute("INSERT OR IGNORE INTO data_sources (name) VALUES ('PulseDB_Vital')")
#     c.execute("INSERT OR IGNORE INTO data_sources (name) VALUES ('PulseDB_MIMIC')")
    
#     c.execute('''CREATE TABLE IF NOT EXISTS mat_files
#                  (id INTEGER PRIMARY KEY, 
#                   data_source_id INTEGER,
#                   filename TEXT,
#                   total_segments INTEGER,
#                   FOREIGN KEY (data_source_id) REFERENCES data_sources(id),
#                   UNIQUE (data_source_id, filename))''')
    
#     c.execute('''CREATE TABLE IF NOT EXISTS segment_annotations
#                  (id INTEGER PRIMARY KEY,
#                   file_id INTEGER,
#                   segment_index INTEGER,
#                   status INTEGER,
#                   max_dtw_score REAL,
#                   FOREIGN KEY (file_id) REFERENCES mat_files(id))''')
    
#     c.execute("CREATE INDEX IF NOT EXISTS idx_file_segments ON segment_annotations(file_id, segment_index)")
    
#     conn.commit()
#     return conn 

# def process_database(conn, data_source, directory):
#     for filename in tqdm(os.listdir(directory)):
#         try:
#             file_path = os.path.join(directory, filename)
#             print(f"Processing {filename}...")
#             process_mat_file(conn, data_source, file_path)
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

# def add_file_annotation(conn, data_source, filename, annotations, max_dtw_scores):
#     c = conn.cursor()
    
#     c.execute("SELECT id FROM data_sources WHERE name = ?", (data_source,))
#     data_source_id = c.fetchone()[0]
    
#     c.execute('''INSERT OR REPLACE INTO mat_files 
#                  (data_source_id, filename, total_segments) 
#                  VALUES (?, ?, ?)''', 
#               (data_source_id, filename, len(annotations)))
#     file_id = c.lastrowid
    
#     for i, (status, max_dtw_score) in enumerate(zip(annotations, max_dtw_scores)):
#         c.execute('''INSERT OR REPLACE INTO segment_annotations 
#                      (file_id, segment_index, status, max_dtw_score) 
#                      VALUES (?, ?, ?, ?)''', 
#                   (file_id, i, status, max_dtw_score))
    
#     conn.commit()
# def process_limited_files(conn, data_source, directory, limit=5):
#     processed = 0
#     for filename in os.listdir(directory):
#         if processed >= limit:
#             break
#         if filename.endswith('.mat'):
#             try:
#                 file_path = os.path.join(directory, filename)
#                 print(f"Processing {filename}...")
#                 process_mat_file(conn, data_source, file_path)
#                 processed += 1
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")


# def get_segment_status(conn, data_source, filename, segment_index):
#     c = conn.cursor()
#     c.execute('''SELECT sa.status, sa.max_dtw_score
#                  FROM segment_annotations sa
#                  JOIN mat_files mf ON sa.file_id = mf.id
#                  JOIN data_sources ds ON mf.data_source_id = ds.id
#                  WHERE ds.name = ? AND mf.filename = ? AND sa.segment_index = ?''',
#               (data_source, filename, segment_index))
#     return c.fetchone()

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS file_annotations
                 (filename TEXT PRIMARY KEY,
                  data_source TEXT,
                  annotations BLOB)''')
    
    conn.commit()
    return conn

def determine_status(max_dtw_score):
    if max_dtw_score <= 0.3:
        return 0  # 正常
    elif max_dtw_score > 0.4:
        return 1  # 异常
    else:
        return 2  # 不确定

def process_segment(args):
    segment, turns = args
    if len(segment) == 0 or len(turns) < 2:
        return 3  # 使用3表示无效数据
    
    dtw_scores = calculate_dtw_scores(segment, turns)
    max_dtw_score = max(dtw_scores) if dtw_scores else 0
    return determine_status(max_dtw_score) if max_dtw_score != 0 else 3

# def process_mat_file(data_source, file_path):
#     try:
#         with h5py.File(file_path, 'r') as f:
#             matdata = f['Subj_Wins']
            
#             ABP_Raws = [f[ref][:].flatten() for ref in matdata['ABP_Raw'][0]]
#             ABP_Turns = [f[ref][:].flatten().astype(int) - 1 for ref in matdata['ABP_Turns'][0]]
            
#         num_segments = len(ABP_Raws)
#         with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#             args = [(ABP_Raws[i], ABP_Turns[i], i) for i in range(num_segments)]
#             results = list(tqdm(pool.imap(process_segment, args), total=num_segments, desc=f"Processing {os.path.basename(file_path)}"))
        
#         annotations = [0] * num_segments
#         for segment_index, status in results:
#             annotations[segment_index] = status
#         return {
#             "filename": os.path.basename(file_path),
#             "data_source": data_source,
#             "annotations": annotations
#         }
#     except Exception as e:
#         logging.error(f"Error processing file {file_path}: {str(e)}")
#         return None
    
# def read_mat_files(file_queue, task_queue):
#     while True:
#         try:
#             file_path = file_queue.get_nowait()
#         except queue.Empty:
#             break

#         try:
#             with h5py.File(file_path, 'r') as f:
#                 matdata = f['Subj_Wins']
#                 ABP_Raws = [f[ref][:].flatten() for ref in matdata['ABP_Raw'][0]]
#                 ABP_Turns = [f[ref][:].flatten().astype(int) - 1 for ref in matdata['ABP_Turns'][0]]

#             for i in range(len(ABP_Raws)):
#                 task_queue.put((ABP_Raws[i], ABP_Turns[i], os.path.basename(file_path), i))
#         except Exception as e:
#             logging.error(f"Error reading file {file_path}: {str(e)}")

#     task_queue.put(None)  # 发送结束信号



# def process_database_parallel(data_source, db_name, limit=10):
#     client = MongoClient('localhost', 57017)
#     db = client[db_name]
#     collection = db[data_source]
    
#     file_paths = [os.path.join(data_source, f) for f in os.listdir(data_source) if f.endswith('.mat')][:limit]
    
#     for file_path in tqdm(file_paths, desc=f"Processing files in {data_source}"):
#         result = process_mat_file(data_source, file_path)
#         if result is not None:
#             try:
#                 collection.update_one(
#                     {"filename": result["filename"]},
#                     {"$set": result},
#                     upsert=True
#                 )
#             except Exception as e:
#                 logging.error(f"Error updating database for file {result['filename']}: {str(e)}")
    
#     client.close()


# def get_mongodb_client(max_retries=3, retry_delay=5):
#     for attempt in range(max_retries):
#         try:
#             client = MongoClient('localhost', 57017, serverSelectionTimeoutMS=5000)
#             client.server_info()  # 这将触发与服务器的实际连接
#             return client
#         except ConnectionFailure as e:
#             print(f"连接失败，尝试 {attempt + 1}/{max_retries}: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(retry_delay)
#     raise ConnectionFailure("无法连接到 MongoDB 服务器")

# def test_query(db_name, data_source):
#     try:
#         client = get_mongodb_client()
#         db = client[db_name]
#         collection = db[data_source]
        
#         # 获取集合中的文档数量
#         doc_count = collection.count_documents({})
#         print(f"Total documents in {data_source}: {doc_count}")
        
#         # 获取第一个文档作为样本
#         sample_doc = collection.find_one()
#         if sample_doc:
#             print("Sample document:")
#             print(f"Filename: {sample_doc.get('filename')}")
#             print(f"Data source: {sample_doc.get('data_source')}")
#             print(f"Number of annotations: {len(sample_doc.get('annotations', []))}")
#         else:
#             print("No documents found in the collection.")
        
#     except Exception as e:
#         print(f"Query failed: {str(e)}")
#     finally:
#         if 'client' in locals():
#             client.close()

# def main():
#     # 处理少量 Vital 数据
#     start_time_vital = time.time()
#     process_database_parallel('PulseDB_Vital', 'pulsedb_annotations_vital_test', limit=5)
#     end_time_vital = time.time()
#     print(f"Processing time for 5 Vital files: {end_time_vital - start_time_vital:.2f} seconds")

#     # 测试查询 Vital 数据
#     test_query('pulsedb_annotations_vital', 'PulseDB_Vital')

#     # 处理少量 MIMIC 数据
#     start_time_mimic = time.time()
#     process_database_parallel('PulseDB_MIMIC', 'pulsedb_annotations_mimic_test', limit=5)
#     end_time_mimic = time.time()
#     print(f"Processing time for 5 MIMIC files: {end_time_mimic - start_time_mimic:.2f} seconds")

#     # 测试查询 MIMIC 数据
#     test_query('pulsedb_annotations_mimic', 'PulseDB_MIMIC')
def process_mat_file(file_path, pool):
    try:
        with h5py.File(file_path, 'r') as f:
            matdata = f['Subj_Wins']
            
            ABP_Raws = [f[ref][:].flatten() for ref in matdata['ABP_Raw'][0]]
            ABP_Turns = [f[ref][:].flatten().astype(int) - 1 for ref in matdata['ABP_Turns'][0]]
        
        # 使用进程池并行处理segments
        results = list(pool.imap(process_segment, zip(ABP_Raws, ABP_Turns)))
        
        return os.path.basename(file_path), results
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return None

def write_to_db(conn, data_source, filename, annotations):
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO file_annotations
                 (filename, data_source, annotations)
                 VALUES (?, ?, ?)''',
              (filename, data_source, sqlite3.Binary(bytes(annotations))))
    conn.commit()

def process_database(data_source, db_name, limit=None):
    conn = create_database(db_name)
    
    file_paths = [os.path.join(data_source, f) for f in os.listdir(data_source) if f.endswith('.mat')]
    if limit:
        file_paths = file_paths[:limit]
    
    # 创建进程池
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for file_path in tqdm(file_paths, desc=f"Processing files in {data_source}"):
            result = process_mat_file(file_path, pool)
            if result:
                filename, annotations = result
                write_to_db(conn, data_source, filename, annotations)
    
    conn.close()

def main():
    process_database('PulseDB_Vital', 'pulsedb_annotations_vital.db')#, limit=10)
    # process_database('PulseDB_MIMIC', 'pulsedb_annotations_test.db', limit=10)

    # 创建两个进程分别处理 PulseDB_Vital 和 PulseDB_MIMIC
    # p1 = multiprocessing.Process(target=process_database_parallel, args=('PulseDB_Vital', 'pulsedb_annotations_vital'))
    # p2 = multiprocessing.Process(target=process_database_parallel, args=('PulseDB_MIMIC', 'pulsedb_annotations_mimic'))

    # p1.start()
    # p2.start()

    # p1.join()
    # p2.join()

# # 主程序
# conn = create_database()

# # 处理 PulseDB_Vital 数据
# process_database(conn, 'PulseDB_Vital', 'PulseDB_Vital')

# # 处理 PulseDB_MIMIC 数据
# process_database(conn, 'PulseDB_MIMIC', 'PulseDB_MIMIC')

# # 查询示例
# status, max_dtw_score = get_segment_status(conn, 'PulseDB_Vital', 'p000001.mat', 0)
# print(f"Segment status: {status}, Max DTW score: {max_dtw_score}")

# conn.close()

# 主程序
# conn = create_database()

# # 处理 PulseDB_Vital 数据（仅处理5个文件）
# process_limited_files(conn, 'PulseDB_Vital', 'PulseDB_Vital', limit=5)

# # 处理 PulseDB_MIMIC 数据（仅处理5个文件）
# process_limited_files(conn, 'PulseDB_MIMIC', 'PulseDB_MIMIC', limit=5)

# # 查询示例
# print("\n查询示例:")
# for data_source in ['PulseDB_Vital', 'PulseDB_MIMIC']:
#     c = conn.cursor()
#     c.execute("SELECT filename FROM mat_files WHERE data_source_id = (SELECT id FROM data_sources WHERE name = ?)", (data_source,))
#     filenames = c.fetchall()
    
#     for (filename,) in filenames:
#         status, max_dtw_score = get_segment_status(conn, data_source, filename, 0)
#         print(f"{data_source} - {filename} - Segment 0 status: {status}, Max DTW score: {max_dtw_score}")

# conn.close()
if __name__ == '__main__':
    main()