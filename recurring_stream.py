import itertools

def recurring_stream_generator(labeled_streams, chunk_size=1000):
    """
    Cycles through labeled streams. 
    labeled_streams: List of tuples like [("Concept_0", stream_obj), ...]
    """
    stream_cycle = itertools.cycle(labeled_streams)
    
    for label, stream in stream_cycle:
        for x, y in stream.take(chunk_size):
            yield x, y, label

