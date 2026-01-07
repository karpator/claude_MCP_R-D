import asyncio, warnings, copy, time
import sys
from typing import AsyncIterable, Tuple, Union, Any, Coroutine

from application_logging import ApplicationLogging

logger = ApplicationLogging.depends()


class BaseNode:
    def __init__(self):
        self.params, self.successors = {}, {}
        self.name = self.get_instance_name() or f"node_{hash(self)}"
        self.flow = None  # Will be set by Flow._propagate_flow
        self.parent = None  # Will be set by Flow._propagate_flow

    def get_instance_name(self):
        """Find the variable name this instance is assigned to, if any"""
        try:
            frame = sys._getframe(1)
            while frame:
                for scope in (frame.f_locals, frame.f_globals):
                    for key, value in scope.items():
                        if value is self and not key.startswith('_') and key != 'self':
                            return key
                frame = frame.f_back
        except (AttributeError, ValueError):
            pass
        return None

    def _get_name(self):
        """Return the instance name, either from name attribute or lookup"""
        return self.name or self.get_instance_name() or f"node_{hash(self)}"

    def set_params(self, params):
        self.params = params

    def next(self, node, action="default"):
        if action in self.successors: warnings.warn(f"Overwriting successor for action '{action}'")
        self.successors[action] = node
        return node

    def prep(self, shared):
        pass

    def exec(self, prep_res):
        pass

    def post(self, shared, prep_res, exec_res):
        pass

    def _exec(self, prep_res):
        """
        Central execution method with logging. Handles both sync and async execution.
        Subclasses should override _do_exec() instead of this method.
        """
        logger().debug(f"Executing node: {self._get_name()} with params: {self.params} and prep_res: {prep_res}")
        result = self._do_exec(prep_res)
        
        # Handle async case
        if asyncio.iscoroutine(result):
            async def async_wrapper():
                final_result = await result  # type: ignore
                logger().debug(f"Node {self._get_name()} execution completed")
                return final_result
            return async_wrapper()
        
        # Handle sync case
        logger().debug(f"Node {self._get_name()} execution completed")
        return result
    
    def _do_exec(self, prep_res):
        """Override this method in subclasses to provide custom execution logic."""
        return self.exec(prep_res)

    def _run(self, shared):
        p = self.prep(shared)
        e = self._exec(p)
        return self.post(shared, p, e)

    def run(self, shared):
        if self.successors: warnings.warn("Node won't run successors. Use Flow.")
        return self._run(shared)

    def __rshift__(self, other):
        return self.next(other)

    def __sub__(self, action):
        if isinstance(action, str): return _ConditionalTransition(self, action)
        raise TypeError("Action must be a string")


class _ConditionalTransition:
    def __init__(self, src, action): self.src, self.action = src, action

    def __rshift__(self, tgt): return self.src.next(tgt, self.action)


class Node(BaseNode):
    def __init__(self, max_retries=1, wait=0):
        super().__init__()
        self.max_retries, self.wait = max_retries, wait

    def exec_fallback(self, prep_res, exc):
        raise exc

    def _do_exec(self, prep_res):
        """Override _do_exec to add retry logic, but still use parent's _exec for logging"""
        for i in range(self.max_retries):
            try:
                return self.exec(prep_res)
            except Exception as e:
                if i == self.max_retries - 1: return self.exec_fallback(prep_res, e)
                if self.wait > 0:
                    time.sleep(self.wait)
        return None


class BatchNode(Node):
    def _exec(self, items): return [super(BatchNode, self)._exec(i) for i in (items or [])]


class Flow(BaseNode):
    def __init__(self, start, name=None):
        super().__init__()
        self.start_node = start
        self.name = name or self.get_instance_name() or f"flow_{hash(self)}"
        self._propagate_flow(self.start_node)

    def _propagate_flow(self, node, visited=None):
        """Set flow and parent references on all nodes in the flow"""
        if visited is None:
            visited = set()

        if node is None or id(node) in visited:
            return

        visited.add(id(node))
        node.flow = self
        node.parent = self.parent if hasattr(self, 'parent') else None

        for successor in node.successors.values():
            self._propagate_flow(successor, visited)

    def start(self, start):
        self.start_node = start
        return start

    def get_next_node(self, curr, action):
        nxt = curr.successors.get(action or "default")
        if not nxt and curr.successors: warnings.warn(f"Flow ends: '{action}' not found in {list(curr.successors)}")
        return nxt

    def _orch(self, shared, params=None):
        curr = copy.copy(self.start_node)
        p = params or {**self.params}
        last_action = None
        while curr:
            curr.set_params(p)
            last_action = curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action

    def _run(self, shared):
        p = self.prep(shared)
        o = self._orch(shared)
        return self.post(shared, p, o)

    def post(self, shared, prep_res, exec_res):
        return exec_res


class BatchFlow(Flow):
    def _run(self, shared):
        pr = self.prep(shared) or []
        for bp in pr: self._orch(shared, {**self.params, **bp})
        return self.post(shared, pr, None)


class AsyncNode(Node):
    async def prep_async(self, shared):
        pass

    async def exec_async(self, prep_res):
        pass

    async def exec_fallback_async(self, prep_res, exc):
        raise exc

    async def post_async(self, shared, prep_res, exec_res):
        pass

    def _do_exec(self, prep_res):
        """Override to return async execution coroutine - parent's _exec will handle logging"""
        return self._do_exec_async(prep_res)

    async def _do_exec_async(self, prep_res):
        """Async retry logic"""
        for i in range(self.max_retries):
            try:
                return await self.exec_async(prep_res)
            except Exception as e:
                if i == self.max_retries - 1:
                    return await self.exec_fallback_async(prep_res, e)
                if self.wait > 0:
                    await asyncio.sleep(self.wait)
        return None

    async def run_async(self, shared):
        if self.successors: warnings.warn("Node won't run successors. Use AsyncFlow.")
        return await self._run_async(shared)

    async def _run_async(self, shared):
        p = await self.prep_async(shared)
        e = await self._exec(p)
        return await self.post_async(shared, p, e)

    def _run(self, shared):
        raise RuntimeError("Use run_async.")


class AsyncBatchNode(AsyncNode, BatchNode):
    async def _exec(self, items):
        """Process batch items, each will be logged by parent's _exec"""
        return [await super(AsyncBatchNode, self)._exec(i) for i in items]


class AsyncParallelBatchNode(AsyncNode, BatchNode):
    async def _exec(self, items):
        """Process batch items in parallel, each will be logged by parent's _exec"""
        return await asyncio.gather(
            *(super(AsyncParallelBatchNode, self)._exec(i) for i in items))


class AsyncFlow(Flow, AsyncNode):
    async def _orch_async(self, shared, params=None):
        curr = copy.copy(self.start_node)
        p = params or {**self.params}
        last_action = None
        while curr:
            curr.set_params(p)
            last_action = await curr._run_async(shared) if isinstance(curr, AsyncNode) else curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action

    async def _run_async(self, shared):
        p = await self.prep_async(shared)
        o = await self._orch_async(shared)
        return await self.post_async(shared, p, o)

    async def post_async(self, shared, prep_res, exec_res): return exec_res


class AsyncBatchFlow(AsyncFlow, BatchFlow):
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        for bp in pr: await self._orch_async(shared, {**self.params, **bp})
        return await self.post_async(shared, pr, None)


class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        await asyncio.gather(*(self._orch_async(shared, {**self.params, **bp}) for bp in pr))
        return await self.post_async(shared, pr, None)


class AsyncGeneratorNode(Node):
    async def prep_async(self, shared):
        pass

    async def exec_async_gen(self, prep_res, context):
        pass

    async def exec_fallback_async(self, prep_res, exc):
        raise exc

    async def post_async_gen(self, shared, prep_res, exec_res, context):
        pass

    async def _exec_gen(self, prep_res, context):
        for i in range(self.max_retries):
            try:
                return self.exec_async_gen(prep_res, context)
            except Exception as e:
                if i == self.max_retries - 1:
                    return await self.exec_fallback_async(prep_res, e)
                if self.wait > 0:
                    await asyncio.sleep(self.wait)
        return None

    async def run_async(self, shared):
        if self.successors: warnings.warn("Node won't run successors. Use AsyncFlow.")
        return self._run_async(shared)

    async def _run_async(self, shared) -> Tuple[Union[AsyncIterable, Any], Coroutine]:
        p = await self.prep_async(shared)
        c = {}
        e = await self._exec_gen(p, c)

        post_coroutine = self.post_async_gen(shared, p, e, c)

        return e, post_coroutine

    def _run(self, shared):
        raise RuntimeError("Use run_async.")


class AsyncGeneratorFlow(AsyncFlow, AsyncGeneratorNode):
    async def _orch_async(self, shared, params=None):
        return await self._orch_async_gen(shared, params)

    async def _orch_async_gen(self, shared, params=None):
        final_last_action = None

        async def generator():
            nonlocal final_last_action
            curr = copy.copy(self.start_node)
            p = params or {**self.params}

            while curr:
                curr.set_params(p)
                if isinstance(curr, AsyncGeneratorNode):
                    result, coro = await curr._run_async(shared)
                    if isinstance(result, AsyncIterable):
                        async for res in result:
                            yield res
                    final_last_action = await coro
                elif isinstance(curr, AsyncNode):
                    final_last_action = await curr._run_async(shared)
                elif isinstance(curr, Node):
                    final_last_action = curr._run(shared)
                curr = copy.copy(self.get_next_node(curr, final_last_action))

        async def get_final_action():
            return final_last_action

        return generator(), get_final_action()

    async def _run_async(self, shared):
        gen, coro = await self._run_async_gen(shared)
        async for item in gen:
            yield item
        await coro

    async def _run_async_gen(self, shared):
        p = await self.prep_async(shared)
        gen, coro = await self._orch_async(shared)

        async def generator():
            async for item in gen:
                yield item

        action = await coro

        post_coroutine = self.post_async(shared, p, action)

        return generator(), post_coroutine

    async def post_async(self, shared, prep_res, exec_res):
        return exec_res

    async def run_async(self, shared):
        if self.successors: warnings.warn("Node won't run successors. Use AsyncFlow.")
        return self._run_async(shared)
