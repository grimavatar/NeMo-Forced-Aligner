def suppress_logging():
    # Suppress standard Python logging warnings
    import logging
    # logging.getLogger().setLevel(logging.ERROR)
    logging.disable(logging.WARNING)

    # # Suppress NeMo's verbose Info (I) and Warning (W) logs
    # import nemo.utils as nemo_utils
    # nemo_utils.logging.setLevel(nemo_utils.logging.ERROR)
