import os
import sys
import json
import boto3
import base64
import time
import numpy as np
from io import BytesIO
from PIL import Image
from facenet_pytorch import MTCNN
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import (
    SubscriptionResponseMessage,
    QOS
)

from datetime import datetime, timedelta, timezone

# Extend path for facenet_pytorch
sys.path.append(os.path.join(os.path.dirname(__file__), 'facenet_pytorch'))

# AWS SQS Queues
SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/861276114572/1230469840-req-queue"
RESPONSE_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/861276114572/1230469840-resp-queue"
sqs = boto3.client("sqs")

# Initialize MTCNN once
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

class FaceDetectionComponent:
    def __init__(self):
        print("[INIT] Initializing FaceDetectionComponent...")
        self.client = GreengrassCoreIPCClientV2()
        self.recent_requests = {}  # ✅ Key: request_id or filename, Value: timestamp
        self.dedup_window = timedelta(seconds=60)  # ✅ 1-minute skip window
        self.mqtt_topic = "clients/1230469840-IoTThing"
        print("[INIT] Initialization complete.")

    def handle_decode(self, content, filename, request_id):
        print(f"[DECODE] Decoding image for filename: {filename}")

        try:
            # Fix base64 padding
            if len(content) % 4:
                content += '=' * (4 - len(content) % 4)

            image_data = base64.b64decode(content)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            print(f"[INFO] Image size: {image.size}")
        except Exception as e:
            print(f"[ERROR] Invalid image data: {str(e)}")
            return
        
        try:
            face, prob = mtcnn(image, return_prob=True)
            if face is None or prob is None or len(face.shape) == 0:
                print("[INFO] No face detected.")
                message_body = {
                    "request_id": request_id,
                    "result": "No-Face"
                }
                response = sqs.send_message(QueueUrl=RESPONSE_QUEUE_URL, MessageBody=json.dumps(message_body))
                print(f"[INFO] Sent 'No-Face' to response queue. MessageId: {response['MessageId']}")
                return

            print(f"[INFO] Face detected. Probability: {prob}")

            # Normalize and convert to image
            face_img = (face - face.min()) / (face.max() - face.min())
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
            face_pil = Image.fromarray(face_img.astype(np.uint8), mode="RGB")

            # Encode to base64
            buffered = BytesIO()
            face_pil.save(buffered, format="JPEG", quality=95)
            face_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Send to SQS
            message_body = {
                "request_id": request_id,
                "filename": filename,
                "face_image": face_base64
            }
            response = sqs.send_message(QueueUrl=SQS_QUEUE_URL, MessageBody=json.dumps(message_body))
            print(f"[INFO] Sent face to request queue. MessageId: {response['MessageId']}")

        except Exception as e:
            print(f"[ERROR] Face detection failed: {str(e)}")

    def subscribe(self):
        print(f"[SUBSCRIBE] Subscribing to topic: {self.mqtt_topic}")

        def on_stream_event(event: SubscriptionResponseMessage):
            #print(f"[DEBUG] Raw payload: {event}")
            try:
                #print(f"event.binary_message: {event.binary_message}")
                #print(f"event.binary_message.message: {event.binary_message.message}")
                #print(f"event.binary_message.message.decode: {event.binary_message.message.decode('utf-8')}")
                payload = event.binary_message.message.decode("utf-8")

                payload = json.loads(payload)

                encoded = payload["encoded"]
                request_id = payload["request_id"]
                filename = payload["filename"]

                now = datetime.now(timezone.utc)

                if encoded and request_id and filename:

                    # Deduplication based on request_id
                    last_time = self.recent_requests.get(request_id)
                    if last_time and now - last_time < self.dedup_window:
                        print(f"[SKIP] Recently processed request_id: {request_id}, skipping.")
                        return
                    self.recent_requests[request_id] = now  # ✅ Store/update timestamp
                    print(f"[EVENT] Processing request_id: {request_id}")
                    self.handle_decode(encoded, filename, request_id)
                else:
                    print("[WARN] Incomplete message payload.")
            except Exception as e:
                print(f"[ERROR] Failed to process message: {str(e)}", file=sys.stderr)

        def on_stream_error(error: Exception):
            print(f"[STREAM ERROR] {str(error)}", file=sys.stderr)
            return True  # Reconnect stream

        def on_stream_closed():
            print("[INFO] Subscription stream closed.")

        try:
            _, op = self.client.subscribe_to_topic(
                topic=self.mqtt_topic,
                on_stream_event=on_stream_event,
                on_stream_error=on_stream_error,
                on_stream_closed=on_stream_closed
            )
            print("[SUBSCRIBE] Subscription established. Waiting for messages...")
            while True:
                time.sleep(1)

        except Exception as e:
            print(f"[SUBSCRIBE] Unauthorized to subscribe to topic: {self.mqtt_topic}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[SUBSCRIBE] Unexpected error: {str(e)}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    print("[MAIN 11] Starting FaceDetectionComponent...")
    component = FaceDetectionComponent()
    component.subscribe()