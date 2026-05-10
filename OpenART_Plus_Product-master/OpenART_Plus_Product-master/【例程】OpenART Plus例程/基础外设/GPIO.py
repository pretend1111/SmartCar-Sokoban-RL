from machine import Pin # 从pyb导入Pin
import time

#初始化引脚
#Pin.OUT 为输出, Pin.IN 为输入, Pin.IN_PUP 为上拉输入, Pin.IN_PDN 为下拉输入
pin0=Pin("M12", Pin.OUT)
pin1=Pin("M11", Pin.OUT)
pin2=Pin("M10", Pin.OUT)
pin3=Pin("M9", Pin.OUT)

pin4=Pin("M4", Pin.OUT)
pin5=Pin("M5", Pin.OUT)
pin6=Pin("J6", Pin.OUT)

pin7=Pin("J25", Pin.OUT)
pin8=Pin("J26", Pin.OUT)
pin9=Pin("J27", Pin.OUT)
pin10=Pin("J28", Pin.OUT)


while(True):
    #pin0.value()如果没有参数会返回引脚电平，有参数1输出高电平或0输出低电平

    # 翻转B0引脚
    if pin0.value():
        pin0.value(0)
    else:
        pin0.value(1)
    # 以上if-else 语句可以等同为以下语句
    # pin0.value(not pin0.value())

    pin1.value(not pin1.value())
    pin2.value(not pin2.value())
    pin3.value(not pin3.value())
    pin4.value(not pin4.value())
    pin5.value(not pin5.value())
    pin6.value(not pin6.value())
    pin7.value(not pin7.value())
    pin8.value(not pin8.value())
    pin9.value(not pin9.value())
    pin10.value(not pin10.value())
    #延时100ms
    time.sleep_ms(100)
