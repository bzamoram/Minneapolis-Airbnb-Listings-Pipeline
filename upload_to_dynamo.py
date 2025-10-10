import boto3
import json
from decimal import Decimal

TABLE_NAME = 'AirbnbListings'

with open('airbnb_listings-reviews.json', encoding='utf-8') as f:
    items = json.load(f, parse_float=Decimal)

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(TABLE_NAME)

BATCH_SIZE = 25
for i in range(0, len(items), BATCH_SIZE):
    batch = items[i:i + BATCH_SIZE]
    with table.batch_writer() as batch_writer:
        for item in batch:
            batch_writer.put_item(Item=item)
print(f"Uploaded {len(items)} items to DynamoDB table '{TABLE_NAME}'")
# upload_to_dynamo.py





