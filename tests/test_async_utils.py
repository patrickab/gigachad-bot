import asyncio
import gc
import weakref

from lib.async_utils import KeyedLocks


async def test_keyed_locks_exclude_per_key_and_forget_unused_keys():
    locks: KeyedLocks[str] = KeyedLocks()
    held = locks("a")
    async with held:
        assert locks("a").locked()
        assert not locks("b").locked()
    forgotten = weakref.ref(held)
    del held
    gc.collect()
    assert forgotten() is None


async def test_waiters_of_one_key_take_turns():
    locks: KeyedLocks[str] = KeyedLocks()
    inside = 0
    peak = 0

    async def hold():
        nonlocal inside, peak
        async with locks("a"):
            inside += 1
            peak = max(peak, inside)
            await asyncio.sleep(0)
            inside -= 1

    await asyncio.gather(hold(), hold(), hold())
    assert peak == 1
