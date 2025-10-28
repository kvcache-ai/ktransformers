from pymongo import MongoClient, errors
import json
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# === 加载 secrets.json 文件 ===
with open(os.path.join(script_dir,"mongo.json")) as f:
    secrets = json.load(f)

MONGO_URI = secrets["mongo_uri"]
DB_NAME = secrets["db_name"]
COLLECTION_NAME = secrets["collection_name"]

# === 连接 MongoDB ===
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# 创建唯一索引（只需执行一次）
collection.create_index(
    [("timestamp", 1), ("test_parameters.CPUInfer_parameter", 1)],
    unique=True
)

# === 插入函数 ===
def insert_jsonl_file(file_path):
    total_inserted = 0
    total_skipped = 0

    with open(file_path, "r") as f:
        docs = [json.loads(line) for line in f if line.strip()]
        try:
            result = collection.insert_many(docs, ordered=False)
            inserted = len(result.inserted_ids)
            total_inserted += inserted
            print(f"[✓] {file_path} 插入 {inserted} 条记录")
        except errors.BulkWriteError as e:
            inserted = len(e.details.get("writeErrors", []))
            skipped = len(docs) - inserted
            total_inserted += inserted
            total_skipped += skipped
            print(f"[!] {file_path} 插入 {inserted} 条，跳过重复后 {skipped} 条")
    
    return total_inserted, total_skipped



insert_jsonl_file( os.path.join(script_dir, "bench_results.jsonl"))
