from __future__ import annotations

import re
from dataclasses import dataclass

from .buffer import HeliosBuffer
from .model import HeliosBlockIndex, SnapshotBlock, TextSpan
from .tokenizer import RegexTokenizer

HEADER_END_MARKER = b"END OF INPUT PARAMETERS"
BLOCK_DELIMITER = b"--------------------------------------------------------------------"
DIAGNOSTICS_MARKER = b"Radiation Cooling Rates and Boundary Fluxes"
CYCLE_HEADER_PATTERN = re.compile(rb"^\s*Cycle\s+Time\s+\(s\)", re.MULTILINE)
# Anchored variant used by the two-stage scan: `bytes.find` locates each
# "Cycle" candidate, this pattern verifies the rest from the line start.
CYCLE_HEADER_ANCHORED = re.compile(rb"\s*Cycle\s+Time\s+\(s\)")
CYCLE_ANCHOR = b"Cycle"


@dataclass(slots=True)
class StructuralIndexer:
    tokenizer: RegexTokenizer

    def build(self, buffer: HeliosBuffer) -> HeliosBlockIndex:
        data = buffer.raw
        header_marker = data.find(HEADER_END_MARKER)
        if header_marker < 0:
            raise ValueError(f"Missing {HEADER_END_MARKER!r} marker in {buffer.source}.")

        header_stop = data.find(b"\n", header_marker)
        if header_stop < 0:
            header_stop = len(buffer)
        else:
            header_stop += 1
        header_span = TextSpan(0, header_stop)

        snapshot_blocks: list[SnapshotBlock] = []
        pending_start: int | None = None
        pending_cycle_stop: int | None = None
        # Two-stage scan. `bytes.find` is a fast substring search, and the
        # anchored pattern verifies the header from its line start. Walking the
        # whole buffer with the MULTILINE pattern costs about 7x more because
        # the `\s*` prefix defeats the regex engine's literal-scan shortcut.
        matches: list[tuple[int, int]] = []
        search_from = header_stop
        while True:
            hit = data.find(CYCLE_ANCHOR, search_from)
            if hit < 0:
                break
            search_from = hit + 1
            line_start = data.rfind(b"\n", 0, hit) + 1
            anchored = CYCLE_HEADER_ANCHORED.match(data, line_start)
            if anchored is None:
                continue
            # `^\s*` may match from an earlier line start inside the same
            # whitespace run; report that position, as the regex did.
            run = line_start
            while run > 0 and data[run - 1] in b" \t\r\n":
                run -= 1
            newline = data.find(b"\n", run, line_start)
            start = newline + 1 if newline >= 0 else line_start
            matches.append((start, anchored.end()))
        for index, (start, match_end) in enumerate(matches):
            cycle_header_stop = data.find(b"\n", match_end)
            if cycle_header_stop < 0:
                cycle_header_stop = len(buffer)
            else:
                cycle_header_stop += 1
            if pending_start is not None and pending_cycle_stop is not None:
                previous_stop = start
                diagnostics_marker = data.find(DIAGNOSTICS_MARKER, pending_start, previous_stop)
                diagnostics_span = TextSpan(diagnostics_marker, previous_stop) if diagnostics_marker >= 0 else None
                snapshot_blocks.append(
                    SnapshotBlock(
                        index=index - 1,
                        span=TextSpan(pending_start, previous_stop),
                        cycle_header_span=TextSpan(pending_start, min(previous_stop, pending_cycle_stop)),
                        diagnostics_span=diagnostics_span,
                    )
                )
            pending_start = start
            pending_cycle_stop = cycle_header_stop

        if pending_start is not None and pending_cycle_stop is not None:
            diagnostics_marker = data.find(DIAGNOSTICS_MARKER, pending_start)
            diagnostics_span = TextSpan(diagnostics_marker, len(buffer)) if diagnostics_marker >= 0 else None
            snapshot_blocks.append(
                SnapshotBlock(
                    index=len(snapshot_blocks),
                    span=TextSpan(pending_start, len(buffer)),
                    cycle_header_span=TextSpan(pending_start, min(len(buffer), pending_cycle_stop)),
                    diagnostics_span=diagnostics_span,
                )
            )

        return HeliosBlockIndex(header_span=header_span, snapshot_blocks=tuple(snapshot_blocks))
