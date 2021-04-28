from redis import StrictRedis
from redis_cache import RedisCache
import dill


def loads(*args):
    return dill.loads(args[0])


def dumps(*args):
    # print(f"dumping {args[0][0][1:]}")
    if isinstance(args[0][0],tuple):
        return dill.dumps(args[0][0][1:])
    # print(f"dumping keci {args[0]}")
    return dill.dumps(args[0])


client = StrictRedis(host="localhost", decode_responses=False)

cache = RedisCache(
    redis_client=client, prefix="flexcoder", serializer=dumps, deserializer=loads
)


# @cache.cache()
# def f(a, b):
#     return torch.rand(5,3)


# node = SearchNode(None, None, None)
# l = [[1, 2, 44]]


# print(f(node, l))
# print(f(node, l))
