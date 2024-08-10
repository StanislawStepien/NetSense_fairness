def deprecated(message):
    def decorator(func):
        func.is_deprecated = True
        func.deprecation_message = message
        return func

    return decorator
