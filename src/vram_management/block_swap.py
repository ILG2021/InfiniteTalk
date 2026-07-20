"""Wan-style transformer block swapping with optional one-ahead prefetch."""

import logging

import torch


class BlockSwapManager:
    """Keep a prefix of blocks on CUDA and stream a suffix through CUDA.

    This follows the block-swap contract used by WanVideoWrapper: blocks are
    moved as complete modules, rather than wrapping every Linear/Norm forward.
    ``blocks_to_swap`` is the number of trailing transformer blocks placed in
    the swap set.
    """

    def __init__(
        self,
        blocks,
        device,
        blocks_to_swap,
        prefetch_blocks=1,
        use_non_blocking=False,
    ):
        self.blocks = blocks
        self.device = torch.device(device)
        self.num_blocks = len(blocks)
        self.blocks_to_swap = max(0, min(int(blocks_to_swap), self.num_blocks))
        self.swap_start = self.num_blocks - self.blocks_to_swap
        self.prefetch_blocks = max(0, int(prefetch_blocks))
        self.use_non_blocking = bool(use_non_blocking)
        self.current_index = None
        self.prefetched = {}
        self.prefetch_events = {}
        self.prefetch_stream = (
            torch.cuda.Stream(device=self.device)
            if self.prefetch_blocks > 0 and self.blocks_to_swap > 0
            else None
        )

    def is_swapped(self, index):
        return index >= self.swap_start

    def prepare(self):
        """Move resident blocks to CUDA and the swap set to CPU once."""
        for index, block in enumerate(self.blocks):
            if self.is_swapped(index):
                block.to("cpu")
            else:
                block.to(self.device)
        self.current_index = None
        self.prefetched.clear()
        self.prefetch_events.clear()

    def _copy_to_cuda(self, index, stream=None):
        block = self.blocks[index]
        if stream is None:
            block.to(self.device, non_blocking=self.use_non_blocking)
            return None
        with torch.cuda.stream(stream):
            block.to(self.device, non_blocking=self.use_non_blocking)
            event = torch.cuda.Event()
            event.record(stream)
        return event

    def _prefetch(self, index):
        if (not self.is_swapped(index) or index in self.prefetched or
                index == self.current_index):
            return
        if self.prefetch_stream is None:
            return
        # The prefetch stream must wait for any previous GPU->CPU copy of this
        # block before reading its CPU storage again.
        self.prefetch_stream.wait_stream(torch.cuda.current_stream(self.device))
        event = self._copy_to_cuda(index, self.prefetch_stream)
        self.prefetched[index] = True
        self.prefetch_events[index] = event

    def _prefetch_ahead(self, index):
        if self.prefetch_blocks <= 0:
            return
        count = 0
        for next_index in range(index + 1, self.num_blocks):
            if not self.is_swapped(next_index):
                continue
            self._prefetch(next_index)
            count += 1
            if count >= self.prefetch_blocks:
                break

    def before_block(self, index):
        if not self.is_swapped(index):
            if self.current_index is not None:
                self.blocks[self.current_index].to(
                    "cpu", non_blocking=self.use_non_blocking)
                self.current_index = None
            return

        if self.current_index is not None and self.current_index != index:
            self.blocks[self.current_index].to(
                "cpu", non_blocking=self.use_non_blocking)
            self.current_index = None

        if index in self.prefetched:
            event = self.prefetch_events.pop(index)
            torch.cuda.current_stream(self.device).wait_event(event)
            self.prefetched.pop(index, None)
        else:
            self._copy_to_cuda(index)
        self.current_index = index
        self._prefetch_ahead(index)

    def after_forward(self):
        if self.prefetch_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(
                self.prefetch_stream)
        # A prefetch stream may have placed several future blocks on CUDA.
        # Wait for it before moving those blocks back; otherwise the next
        # iteration can race a GPU->CPU copy with the prefetch copy.
        for index in list(self.prefetched):
            self.blocks[index].to("cpu", non_blocking=self.use_non_blocking)
        if self.current_index is not None:
            self.blocks[self.current_index].to(
                "cpu", non_blocking=self.use_non_blocking)
            self.current_index = None
        self.prefetched.clear()
        self.prefetch_events.clear()

    def log_configuration(self):
        logging.info(
            "BlockSwap: %d/%d trailing blocks swapped, prefetch=%d, "
            "non_blocking=%s",
            self.blocks_to_swap,
            self.num_blocks,
            self.prefetch_blocks,
            self.use_non_blocking,
        )
