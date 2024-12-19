import struct
import serial
class IMU():
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB0', 1500000, timeout=1)
        self.pattern = "aa55"

    # 定义转换函数
    def hex_to_float(self,hex_str):
        return struct.unpack('<f', bytes.fromhex(hex_str))[0]
    def get_root_state(self):
        data = self.ser.read(1024)
        hex_str = data.hex()
        first_index = hex_str.find(self.pattern)
        if first_index != -1:
            # 从第一个模式之后继续查找第二个模式
            second_index = hex_str.find(self.pattern, first_index + len(self.pattern))
            if second_index != -1:
                # 截取从第一个到第二个模式之间的数据
                extracted_data = hex_str[first_index:second_index]
                # 查找"aa55"的位置，这里假设"aa55"在数据的开始处
                start_index = extracted_data.find(self.pattern) + 4  # 从"aa55"之后开始截取
                # 截取"aa55"之后的数据
                post_aa55_data = extracted_data[start_index:]
                # 初始化结果列表
                result = []
                # 每四个字符截取一次，截取两次
                for _ in range(2):
                    if len(post_aa55_data) >= 4:
                        result.append(post_aa55_data[:4])
                        post_aa55_data = post_aa55_data[4:]
                    else:
                        break  # 如果不足四个字符，就停止截取
                # 每八个字符截取一次，直到没有剩余数据
                while len(post_aa55_data) >= 8:
                    result.append(post_aa55_data[:8])
                    post_aa55_data = post_aa55_data[8:]
                # 如果还有剩余数据，添加到结果列表
                if post_aa55_data:
                    result.append(post_aa55_data)
                # 将交换后的十六进制数转换为十进制数
                float_values = [self.hex_to_float(hex_str) for hex_str in result[2:]]
                root_state=[float_values[2],float_values[1],float_values[3],float_values[7],float_values[8],float_values[9]]
                return root_state
            else:
                print("没有找到第二个模式")
        else:
            print("模式未找到")
'''
# 分开打印每个部分
part_1 = decimal_value1
part_2 = decimal_value2
part_3 = float_values[0]
part_4 = float_values[1]
part_5 = float_values[2]
part_6 = float_values[3]
part_7 = float_values[4]
part_8 = float_values[5]
part_9 = float_values[6]
part_10 = float_values[7]
part_11 = float_values[8]
part_12 = float_values[9]
part_13 = float_values[10]
part_14 = float_values[11]
#
# # 打印每个部分
print("ID:", part_1)
print("长度:", part_2)
print("时间标:", part_3)
print("俯仰角:", part_4, "°")
print("横滚角:", part_5, "°")
print("航向角:", part_6, "°")
print("X轴加速度:", part_7, "g")
print("Y轴加速度:", part_8, "g")
print("Z轴加速度:", part_9, "g")
print("X轴角速度:", part_10, "°/s")
print("Y轴角速度:", part_11, "°/s")
print("Z轴角速度:", part_12, "°/s")
print("IMU芯片温度", part_13, "℃")
print("crc32校验:", part_14)
'''