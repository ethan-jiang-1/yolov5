import os
import signal


class InterruptSignal(object):
    sig_num = None

    @classmethod
    def get_received_signal(cls):
        return cls.sig_num

    @classmethod
    def reset_kill_signal(cls):
        cls.sig_num = None
        print("reset sig_num")

    @classmethod
    def prompt_kill_signal(cls):
        pid = os.getpid()
        print()
        print('#m# start PID: {}, run following'.format(pid))
        print("")
        print("To Stop Training:")
        print("  kill -n {} {}".format(signal.SIGUSR1, pid)) 
        print()
        print("To Inc Training epochs:")
        print("  kill -n {} {}".format(signal.SIGUSR2, pid))
        print() 


def receive_signal_user(signum, stack):
    InterruptSignal.sig_num = signum
    print()
    print('#r# Received Interrupt Signal: {} '.format(signum))
    print()


signal.signal(signal.SIGUSR1, receive_signal_user)
signal.signal(signal.SIGUSR2, receive_signal_user)
