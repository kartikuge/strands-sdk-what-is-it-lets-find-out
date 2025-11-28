import logging

# root SDK logs
logging.getLogger("strands").setLevel(logging.DEBUG)
logging.getLogger("strands.agent").setLevel(logging.DEBUG)
logging.getLogger("strands.tools").setLevel(logging.DEBUG)
logging.getLogger("strands.events").setLevel(logging.DEBUG)

# global logging formatter
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)