from vlmeval.config import supported_VLM
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
model = supported_VLM['Awaker2.5-VL']()
# 前向单张图片
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # 这张图片上有一个带叶子的红苹果
# 前向多张图片
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # 提供的图片中有两个苹果