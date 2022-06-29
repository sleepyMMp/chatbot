import time, logging, random, requests
from numpy import block
from queue import Queue
from WechatPCAPI import WechatPCAPI
import seq2seq

logging.basicConfig(level=logging.INFO)
queue_recived_msg = Queue()

def on_message(msg):
    queue_recived_msg.put(msg)


def reply_msg(receive_msg):
    return seq2seq.chat(receive_msg)[0]


def login():
    pre_msg = ""
    wx_inst = WechatPCAPI(on_message=on_message,log=logging)
    wx_inst.start_wechat(block=True)
    while not wx_inst.get_myself():
        time.sleep(5)
    print("success")
    while True:
        msg = queue_recived_msg.get()
        if 'msg::single' in msg.get('type'):
            data = msg.get('data')
            if data.get('is_recv', False):
                msgFromInfo = data.get('msgfrominfo')
                if msgFromInfo is not None:
                    wx_id = msgFromInfo.get('wx_id')
                    received_msg = str(data.get('msgcontent'))
                    reply = reply_msg(received_msg)
                    wx_inst.send_text(to_user=wx_id, msg=reply)

if __name__ == '__main__':
    login()