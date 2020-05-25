from facelab.util.lazy import LazyCall


class Test(LazyCall):

    @staticmethod
    def lazy_first():
        print('first')

    @staticmethod
    def lazy_second():
        print('second')


if __name__ == '__main__':
    test = Test()
    test.first()
    test.second()
    test.lazy_call()

