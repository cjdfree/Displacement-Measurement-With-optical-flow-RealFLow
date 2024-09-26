import functools


def print_function_name(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Running function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


# 通用的装饰器函数
def print_function_name_decorator(func):
    return print_function_name(func)
