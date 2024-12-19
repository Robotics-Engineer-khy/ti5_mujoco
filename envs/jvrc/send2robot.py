from ctypes import *
import numpy as np
import struct
import time
VCI_USBCAN2 = 4
STATUS_OK = 1

class VCI_INIT_CONFIG(Structure):
    _fields_ = [("AccCode", c_uint),
                ("AccMask", c_uint),
                ("Reserved", c_uint),
                ("Filter", c_ubyte),
                ("Timing0", c_ubyte),
                ("Timing1", c_ubyte),
                ("Mode", c_ubyte)
                ]

class VCI_CAN_OBJ(Structure):
    _fields_ = [("ID", c_uint),
                ("TimeStamp", c_uint),
                ("TimeFlag", c_ubyte),
                ("SendType", c_ubyte),
                ("RemoteFlag", c_ubyte),
                ("ExternFlag", c_ubyte),
                ("DataLen", c_ubyte),
                ("Data", c_ubyte * 8)
                ]
class VCI_CAN_OBJ_ARRAY(Structure):
    _fields_ = [('SIZE', c_uint16), ('STRUCT_ARRAY', POINTER(VCI_CAN_OBJ))]
    def __init__(self,num_of_structs):
                                                                 #这个括号不能少
        self.STRUCT_ARRAY = cast((VCI_CAN_OBJ * num_of_structs)(),POINTER(VCI_CAN_OBJ))#结构体数组
        self.SIZE = num_of_structs#结构体长度
        self.ADDR = self.STRUCT_ARRAY[0]#结构体数组地址  byref()转c地址
class Controller():
    def __init__(self, nu,gear,motor_pos,kp, kd):
        self.nu = nu
        self.gear = gear[:self.nu]
        self.kt = np.array([0.0754, 0.0754, 0.098, 0.098, 0.0686, 0.049,
                            0.0754, 0.0754, 0.098, 0.098, 0.0686, 0.049])
        self.ubyte_array = c_ubyte * 8
        self.rx_obj = VCI_CAN_OBJ_ARRAY(100)  # 结构体数组
        self.canDLL = cdll.LoadLibrary('./libcontrolcan.so')
        self.can_id = 0
        ret = self.canDLL.VCI_OpenDevice(VCI_USBCAN2, 0, 0)
        if ret != STATUS_OK:
            print('调用 VCI_OpenDevice出错\r\n')
        vci_initconfig = VCI_INIT_CONFIG(0x80000008, 0xFFFFFFFF, 0, 0, 0x00, 0x14, 0)  # 波特率1M
        # 初始通道
        ret = self.canDLL.VCI_InitCAN(VCI_USBCAN2, 0, self.can_id, byref(vci_initconfig))
        if ret != STATUS_OK:
            print('调用 VCI_InitCAN出错\r\n')
        ret = self.canDLL.VCI_StartCAN(VCI_USBCAN2, 0, self.can_id)
        if ret != STATUS_OK:
            print('调用 VCI_StartCAN出错\r\n')

        self.set_pdkt(kp[:self.nu],kd[:self.nu])
        self.set_pos_target(motor_pos[:self.nu])
        time.sleep(3)
        #self.get_motor_state()#****************************************************************************


    def get_motor_state(self):
        for k in range(5):
            data = self.ubyte_array(65, 0, 0, 0, 0, 0, 0, 0)

            for i in range(self.nu):
                vci_can_obj = VCI_CAN_OBJ(i + 1, 0, 0, 0, 0, 0, 1, data)
                ret = 0
                while ret < 1:
                    ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.can_id, byref(vci_can_obj), 1)
            ret = 0
            while ret < 1:
                ret = self.canDLL.VCI_Receive(VCI_USBCAN2, 0, self.can_id, byref(self.rx_obj.ADDR), 2500, 0)
            pos = np.zeros(self.nu)
            vel = np.zeros(self.nu)
            tau = np.zeros(self.nu)
            ids = np.ones(self.nu)
            for i in range(ret):
                if self.rx_obj.STRUCT_ARRAY[i].Data[0] != 65:
                    rx_id=self.rx_obj.STRUCT_ARRAY[i].ID
                    if rx_id<=self.nu:
                        data = self.rx_obj.STRUCT_ARRAY[i].Data
                        ids[rx_id-1] -= 1
                        tau[rx_id-1] = struct.unpack_from('<h', data, 0)[0]  # 使用小端模式('<H')解析前两个字节
                        vel[rx_id-1] = struct.unpack_from('<h', data, 2)[0]  # 使用小端模式('<H')解析第三、四个字节
                        pos[rx_id-1] = struct.unpack_from('<i', data, 4)[0]  # 使用小端模式('<I')解析第五到八字节
            if np.all(ids == 0):
                break

            '''
            for i in range(self.nu):
                #if i in [0, 3, 6, 8]:
                vci_can_obj = VCI_CAN_OBJ(i + 1, 0, 0, 0, 0, 0, 1, data)
                ret = 0
                while ret < 1:
                    ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.can_id, byref(vci_can_obj), 1)
            ret = 0
            while ret < 1:
                ret = self.canDLL.VCI_Receive(VCI_USBCAN2, 0, self.can_id, byref(self.rx_obj.ADDR), 2500, 0)

            pos = np.zeros(self.nu)
            vel = np.zeros(self.nu)
            tau = np.zeros(self.nu)
            ids = np.ones(self.nu)
            for i in range(ret):
                rx_id = self.rx_obj.STRUCT_ARRAY[i].ID
                #if rx_id in [1, 4, 7, 9]:
                data = self.rx_obj.STRUCT_ARRAY[i].Data
                ids[rx_id - 1] -= 1
                tau[rx_id - 1] = struct.unpack_from('<h', data, 0)[0]  # 使用小端模式('<H')解析前两个字节
                vel[rx_id - 1] = struct.unpack_from('<h', data, 2)[0]  # 使用小端模式('<H')解析第三、四个字节
                pos[rx_id - 1] = struct.unpack_from('<i', data, 4)[0]  # 使用小端模式('<I')解析第五到八字节
            if ids.sum() == 0:
                break
            '''
        #print(k)
        tau *= self.kt * self. gear/ 1000.0
        vel *= 0.01 * 2 * np.pi / self.gear
        pos *= 2 * np.pi / 65536.0 / self.gear
        return pos, vel, tau

    def set_pdkt(self, kp, kv):
        send_kp = kp.astype(np.float32)
        send_kv = kv.astype(np.float32)
        send_kt = self.kt.astype(np.float32)
        total_ret=0
        for i in range(self.nu):
            #if i in [0, 3, 6, 8]:
            c_p = struct.pack('<f', send_kp[i])
            data = self.ubyte_array(0x42, 0x20, *c_p, 0, 0)
            vci_can_obj = VCI_CAN_OBJ(i + 1, 0, 0, 0, 0, 0, 6, data)
            ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.can_id, byref(vci_can_obj), 1)
            total_ret+=ret
            c_n = struct.pack('<f', send_kv[i])
            data = self.ubyte_array(0x43, 0x20, *c_n, 0, 0)
            vci_can_obj = VCI_CAN_OBJ(i + 1, 0, 0, 0, 0, 0, 6, data)
            ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.can_id, byref(vci_can_obj), 1)
            total_ret+=ret
            c_k = struct.pack('<f', send_kt[i])
            data = self.ubyte_array(0x45, 0x20, *c_k, 0, 0)
            vci_can_obj = VCI_CAN_OBJ(i + 1, 0, 0, 0, 0, 0, 6, data)
            ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.can_id, byref(vci_can_obj), 1)
            #print("kt_ret: ",ret)
            total_ret += ret
        if total_ret < 3*self.nu:
        #if total_ret < 12:
            print('kp、kv、kt设置失败')

    def set_pos_target(self, target):
        total_ret=0
        send_target = (target * 65536 * self.gear / 2 / np.pi).astype(np.int32)
        for i in range(self.nu):
            #if i in [0, 3, 6, 8]:
            position = struct.pack('<i', send_target[i])
            data = self.ubyte_array(0x1e, *position, 0, 0, 0)
            vci_can_obj = VCI_CAN_OBJ(i + 1, 0, 0, 0, 0, 0, 5, data)  # 单次发送
            ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.can_id, byref(vci_can_obj), 1)
            total_ret += ret
        for i in range(3):
            position = struct.pack('<i', 0)
            data = self.ubyte_array(0x1e, *position, 0, 0, 0)
            vci_can_obj = VCI_CAN_OBJ(self.nu + i + 1, 0, 0, 0, 0, 0, 5, data)  # 单次发送
            ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.can_id, byref(vci_can_obj), 1)
            total_ret += ret
        if total_ret < self.nu+3:
            print('pos设置失败')
    def set_pd_target(self, target):
        total_ret = 0
        send_target = target
        for i in range(self.nu):
            #if i in [0, 3, 6, 8]:
            position = struct.pack('<f', send_target[i])
            data = self.ubyte_array(0x5a, *position, 0, 0, 0)
            vci_can_obj = VCI_CAN_OBJ(i + 1, 0, 0, 0, 0, 0, 5, data)  # 单次发送
            ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.can_id, byref(vci_can_obj), 1)
            total_ret += ret




