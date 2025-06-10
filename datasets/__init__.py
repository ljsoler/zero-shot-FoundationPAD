from .oulu import OULU, OULUTest
from .swim_v2 import SwimV2, SwimV2Test
from .casia_fasd import CASIA_FASD
from .testDatabase import TestDatabase
from .AggregateDB import AggregateDB

DATASET = {
    "OULU": OULU,
    "SWIM": SwimV2,
    "CASIA": CASIA_FASD,
    "Test": TestDatabase,
    "AGGREGATEDB": AggregateDB
}