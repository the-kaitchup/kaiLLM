"""Stage target taps and replay DSpine's learned projection on bounded prompt chunks."""
import torch


class AuxiliaryProjectionGraph:
    def __init__(self, proposer, chunk_size=256):
        self.proposer = proposer
        self.size = min(chunk_size, proposer.max_num_tokens)
        # The query graph owns the first rows. The spill graph has separate
        # storage and cannot overwrite that query's staged input.
        self.inputs = proposer._aux_staging.new_zeros((self.size, proposer._aux_staging.shape[-1]))
        self.graph = None
        self.outputs = None

    def capture(self):
        model = self.proposer.model
        with torch.cuda.device(self.inputs.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                model.combine_hidden_states(self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.outputs = model.combine_hidden_states(self.inputs)
        self.graph = graph

    def stage(self, aux, start, stop, destination):
        offset = 0
        for tap in aux:
            width = tap.shape[-1]
            destination[:stop-start, offset:offset+width].copy_(tap[start:stop])
            offset += width
        if offset != destination.shape[-1]:
            raise ValueError('Unexpected DSpine target tap width')

    def replay(self, aux, start, end, observer):
        if self.graph is None:
            observer.draft_fallback('DSpine auxiliary projection has no captured graph')
        for offset in range(start, end, self.size):
            stop = min(offset + self.size, end)
            self.stage(aux, offset, stop, self.inputs)
            self.graph.replay()
            observer.draft_replayed()
            # The projection is row independent; discard the padded tail.
            self.proposer.hidden_states[offset:stop].copy_(self.outputs[:stop-offset])
