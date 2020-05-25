import collections
from functools import wraps


class DecoRegister(object):
    """装饰器注册器"""

    _register = []

    @classmethod
    def register(cls, cls_name, fn_name):
        cls._register.append((cls_name, fn_name))

    @classmethod
    def is_register(cls, cls_name, fn_name):
        if (cls_name, fn_name) in cls._register:
            return True
        else:
            return False


class FuncBridge(object):
    """调用方法时，实际在使用该类进行调用的方法的注册"""

    unique_obj_cache = {}

    def __init__(self, calling, fn_name, fn_desc=None):
        self.nlpsc = None
        self._calling = calling
        self._fn_name = fn_name
        self._fn_args = ()
        self._fn_kwargs = {}
        self._fn_desc = fn_desc if fn_desc else '{} processing'.format(self._fn_name)
        # fn_return_type = getattr(self._calling, self._fn_name).__annotations__.get('return')
        # fn_return_type_name = fn_return_type.__class__.__name__

        self._fn_return_obj = None
        # if fn_return_type_name not in self.unique_obj_cache.keys():
        #     if fn_return_type:
        #         self._fn_return_obj = fn_return_type()
        #         self.unique_obj_cache[fn_return_type_name] = self._fn_return_obj
        # else:
        #     self._fn_return_obj = self.unique_obj_cache[fn_return_type_name]

    def __call__(self, *args, **kwargs):
        self._fn_args = args
        self._fn_kwargs = kwargs
        return self._fn_return_obj

    def __str__(self):
        return '<FuncBridge {} {} {} {}>'.format(self._calling, self._fn_name, self._fn_args, self._fn_kwargs)

    def call(self, add_bridge=False):
        if add_bridge:
            return getattr(self._calling, self._fn_name)(self, *self._fn_args, **self._fn_kwargs)
        else:
            return getattr(self._calling, self._fn_name)(*self._fn_args, **self._fn_kwargs)

    @property
    def return_obj(self):
        return self._fn_return_obj

    @return_obj.setter
    def return_obj(self, obj):
        self._fn_return_obj = obj

    @property
    def cls_name(self):
        return self._calling.__class__.__name__

    @property
    def fn_name(self):
        return self._fn_name

    @property
    def fn_args(self):
        return self._fn_args

    @fn_args.setter
    def fn_args(self, args):
        self._fn_args = args


def _sugar_lazy_call(fn_name):
    """懒加载方法转调用规则"""

    return 'lazy_{}'.format(fn_name)


def _sugar_magic_call(cls_name, fn_name):
    """系统内建方法转调用规则"""

    return '_{}{}'.format(cls_name, fn_name)


def callable_register(fn):
    """函数调用注册器
    用于把用户调用的方法转换成bridge，lazy_call时使用
    """

    def is_function_valid(cls, func):
        if fn not in dir(cls):
            return True
        else:
            print("AttributeError: '{}' object has no attribute '{}'".format(cls.__class__.__name__,
                                                                             func))
            raise AttributeError

    @wraps(fn)
    def wrapper(obj, fn_name):
        # 调用python内建方法
        if fn_name.startswith('__') or fn_name.endswith('__'):
            raise AttributeError
        elif isinstance(obj, LazyCall):
            # 如果被调用的方法在定义类中已经存在,则不需要进行方法名的转换
            # 调用框架定义方法
            if fn_name.startswith('__'):
                # 系统内置方法和框架内置方法，立即执行
                real_fn_name = _sugar_magic_call(obj.__class__.__name__, fn_name)
                return getattr(obj, real_fn_name)()
            elif fn_name.startswith('_'):
                raise AttributeError
            else:
                # 懒加载执行的方法
                real_fn_name = _sugar_lazy_call(fn_name)
                if is_function_valid(obj, real_fn_name):
                    bridge = FuncBridge(obj, real_fn_name)
                    obj.add_bridge(bridge)
                    bridge.return_obj = obj
                    return bridge
        else:
            print("'@callable_register' decorate 'LazyCall' object")
            raise AttributeError

    return wrapper


class _BridgeChain(object):

    def __init__(self):
        self._chains = collections.OrderedDict()

    def register(self, bridge):
        self._chains[bridge.fn_name] = bridge

    def get_last_bridge(self, bridge, last=1):
        pass

    def get_next_bridge(self, bridge, nex=1):
        pass

    def iter_previous_bridges(self, bridge):
        """返回当前bridge之前bridge的迭代器"""

        ks = list(self._chains.keys())
        p = ks.index(bridge.fn_name)
        for k in ks[:p]:
            yield self._chains[k]

    def iter(self):
        for bridge in self._chains.values():
            yield bridge

    @property
    def bridges(self):
        return self._chains.values()


class LazyCall(object):

    def __init__(self):
        self.call_stack = []
        self.child_obj = None
        self.chain = _BridgeChain()

    @callable_register
    def __getattr__(self, fn):
        return self

    def deepdive(self, obj=None):
        """深入查看当前的调用链，返回的调用链都是bridge

        如果obj为空，则从当前对象开始遍历"""

        if not obj:
            obj = self

        if isinstance(obj, LazyCall):
            for bridge in obj.call_stack:
                self.chain.register(bridge)
                yield bridge

            for bridge in obj.call_stack:
                if bridge.return_obj and bridge.return_obj is not obj:

                    yield from self.deepdive(bridge.return_obj)
        else:
            print("'deepdive' dive in 'NLPShortcutCore' object")
            raise AttributeError

    def add_bridge(self, bridge):
        self.call_stack.append(bridge)

    def iter_bridge(self):
        yield from self.chain.iter()

    @property
    def bridges(self):
        return self.call_stack

    def lazy_call(self):
        """真实执行操作的方法"""
        for bridge in self.deepdive(self):
            bridge.call()



