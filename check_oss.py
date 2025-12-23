import boto3
import os

ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "01989CF435517971960CFB24AAE04E3B") # 填入你的 ID
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "01989CF4355179609DADC82EC3B04ADD") # 填入你的 Secret
ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn")

print(f"🔍 正在尝试连接 OSS...")
print(f"   Endpoint: {ENDPOINT}")
print(f"   AccessKey: {ACCESS_KEY[:6]}******")

try:
    # 初始化客户端
    s3 = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT
    )
    
    # 动作 1: 列出所有 Buckets (最基础的权限测试)
    print("\n⏳ 正在列出 Buckets...")
    response = s3.list_buckets()
    
    print("✅ 连接成功！您的账户下有以下 Buckets:")
    for bucket in response['Buckets']:
        print(f"   - {bucket['Name']}")

    # 动作 2: (可选) 尝试访问某个具体 Bucket 里的文件
    # 如果你知道训练数据所在的 bucket 名字，可以在这里填
    # target_bucket = "wan-video-data" 
    # s3.list_objects_v2(Bucket=target_bucket, MaxKeys=5)
    # print(f"\n✅ 成功访问 Bucket: {target_bucket}")

except Exception as e:
    print("\n❌ 连接失败！请检查以下原因：")
    print(f"1. 秘钥是否正确？")
    print(f"2. Endpoint 是否匹配（北京/杭州/上海）？")
    print(f"3. 服务器是否有外网权限？")
    print(f"详细报错信息: {e}")