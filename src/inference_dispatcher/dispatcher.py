import os, json, time, requests
from confluent_kafka import Consumer, Producer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOP_IN  = os.getenv("TOP_IN", "media.image_uploaded")
TOP_OUT = os.getenv("TOP_OUT", "ml.inference_done")
MODEL_URL = os.getenv("MODEL_URL", "http://rest-api:8800/predict")

c = Consumer({
    "bootstrap.servers": BOOTSTRAP,
    "group.id": "inference-dispatcher",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False
})
p = Producer({"bootstrap.servers": BOOTSTRAP, "enable.idempotence": True})

def call_model(bucket: str, key: str):
    r = requests.post(MODEL_URL, json={"bucket": bucket, "key": key}, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    print("--> Dispatcher script started, subscribing to topic:", TOP_IN)
    c.subscribe([TOP_IN])
    try:
        while True:
            msg = c.poll(1.0)
            if not msg:
                print("--> Waiting for message...")
                continue
            if msg.error():
                print("Kafka error:", msg.error())
                continue

            try:
                evt = json.loads(msg.value())
                # MinIO sends S3-compatible events
                rec = evt["Records"][0] if "Records" in evt else evt
                bucket = rec["s3"]["bucket"]["name"] if "s3" in rec else rec["bucket"]
                key    = rec["s3"]["object"]["key"] if "s3" in rec else rec["key"]
                etag   = rec["s3"]["object"].get("eTag") if "s3" in rec else rec.get("eTag")

                result = call_model(bucket, key)
                out = {
                    "bucket": bucket, "key": key, "eTag": etag,
                    "status": "OK",
                    "results": result,
                    "ts": int(time.time()*1000),
                }
                p.produce(TOP_OUT, key=f"{bucket}/{key}", value=json.dumps(out))
                print(f"--> Successfully processed and produced message for key: {key}")
                c.commit(msg)  # commit only after success
            except Exception as e:
                # keep it simple: log and retry (don’t commit)
                print("Dispatch error:", repr(e))
    finally:
        c.close()

if __name__ == "__main__":
    main()