import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("urbanfl.event_bus")

class EventType(Enum):
    """Core taxonomy of events across the UrbanCityOS ecosystem."""
    CITIZEN_REPORT = auto()
    ALERT_VERIFIED = auto()
    INFRASTRUCTURE_UPDATE = auto()
    DISPATCH_TICK = auto()
    HOSPITAL_STATUS = auto()
    ENVIRONMENTAL_RISK = auto()
    SILO_HEARTBEAT = auto()
    SYSTEM_ALERT = auto()

@dataclass
class Event:
    """Immutable data container for global state exchange."""
    event_type: EventType
    source_silo: int
    data: Dict[str, Any]
    priority: float = 0.5  # 0.0 (lowest) to 1.0 (critical)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        # Enforce constraints
        self.priority = max(0.0, min(1.0, self.priority))

class EventBus:
    """
    Thread-safe, async-capable pub/sub messaging hub connecting all silos and dashboards.
    Implements graceful fallback and rudimentary backpressure via max_queue_size.
    """
    def __init__(self, max_queue_size: int = 5000):
        self.max_queue_size = max_queue_size
        self._subscribers: Dict[EventType, List[Callable[[Event], Any]]] = {
            et: [] for et in EventType
        }
        self._async_subscribers: Dict[EventType, List[Callable[[Event], Any]]] = {
            et: [] for et in EventType
        }
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._loop_task: Optional[asyncio.Task] = None
        self._metrics = {"published": 0, "dropped": 0, "dispatched": 0}

    def subscribe(self, event_type: EventType, callback: Callable[[Event], Any], is_async: bool = False):
        """Register a synchronous or asynchronous listener."""
        if not isinstance(event_type, EventType):
            logger.warning(f"Invalid event_type {event_type} subscription attempt.")
            return
            
        if is_async:
            self._async_subscribers[event_type].append(callback)
        else:
            self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed {callback.__name__} to {event_type.name}")

    def publish(self, event: Event):
        """Immediately dispatch synchronously and enqueue for async processors."""
        self._metrics["published"] += 1
        
        # 1. Execute synchronous callbacks immediately (prevents simulation lag)
        for cb in self._subscribers.get(event.event_type, []):
            try:
                cb(event)
                self._metrics["dispatched"] += 1
            except Exception as e:
                logger.error(f"Sync callback error in {cb.__name__} for {event.event_type.name}: {e}", exc_info=True)
                
        # 2. Queue for async tasks
        if self._async_subscribers.get(event.event_type):
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self._metrics["dropped"] += 1
                logger.warning(f"EventBus queue full ({self.max_queue_size}). Dropping {event.event_type.name}.")

    async def _async_dispatch_loop(self):
        """Background loop for async callbacks."""
        while True:
            event = await self._queue.get()
            try:
                tasks = [
                    asyncio.create_task(cb(event))
                    for cb in self._async_subscribers.get(event.event_type, [])
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    self._metrics["dispatched"] += len(tasks)
            except Exception as e:
                logger.error(f"Async dispatch error: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    def start_async_loop(self):
        """Start the background async dispatch loop strictly within the current event loop."""
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._async_dispatch_loop())

    def stop_async_loop(self):
        if self._loop_task:
            self._loop_task.cancel()
            self._loop_task = None


def setup_default_subscriptions(bus: EventBus):
    """
    Central bootstrap for critical listeners.
    Currently acts as a placeholder as dynamic silos self-register during init,
    but retained for architectural dependency injection compatibility in `state.py`.
    """
    def _catch_alert(e: Event):
        if e.priority > 0.9:
            logger.warning(f"CRITICAL SYSTEM ALERT from silo {e.source_silo}: {e.data}")
            
    bus.subscribe(EventType.SYSTEM_ALERT, _catch_alert)
    bus.subscribe(EventType.CITIZEN_REPORT, lambda e: None)  # Silence standard reports
    logger.info("EventBus core subscriptions initialized.")
