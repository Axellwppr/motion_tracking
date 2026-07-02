import linuxfd
import select


class DictToClass:
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)


class Timer(object):
    '''Timer class for accurate loop rate control
    This class does not use Python's built-in thread timing control
    or management. Only use this class on Linux platforms.
    '''
    def __init__(self, interval: float) -> None:
        self.__epl, self.__tfd = self.__create_timerfd(interval)

    @staticmethod
    def __create_timerfd(interval: float):
        '''Produces a timerfd file descriptor from the kernel
        '''
        tfd = linuxfd.timerfd(rtc=True, nonBlocking=True)
        tfd.settime(interval, interval)
        epl = select.epoll()
        epl.register(tfd.fileno(), select.EPOLLIN)
        return epl, tfd

    def sleep(self) -> None:
        '''Blocks the thread holding this func until the next time point
        '''
        events = self.__epl.poll(-1)
        for fd, event in events:
            if fd == self.__tfd.fileno() and event & select.EPOLLIN:
                self.__tfd.read()
