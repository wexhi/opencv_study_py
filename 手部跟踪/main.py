import random


res_list = []
for i in range(1, 520):
    result = random.randint(1, 520)
    if result > 250:
        res_list.append(result)

if len(res_list) > 260:
    print("聊天")
print(len(res_list))
