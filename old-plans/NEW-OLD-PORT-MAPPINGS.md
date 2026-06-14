# Port rationalization — old → new mapping

Performed 2026-06-14.  All active project files were updated; this document
records the mapping for historical reference (e.g. when reading old-plans/).

## Public ports (host-accessible)

| Old | New  | Service            | Role                              |
|-----|------|--------------------|-----------------------------------|
| 9797 | 8700 | `granite-base`     | Base model proxy (llama.cpp)      |
| 8001 | 8701 | `granite-plus-proxy` | Plus model proxy (chunking/stitching) |
| 8002 | 8702 | `granite-nar`      | NAR model server                  |

## Internal ports (loopback only)

| Old   | New   | Service          | Role                              |
|-------|-------|------------------|-----------------------------------|
| 19797 | 18700 | `llama-server`   | Base model backend (llama.cpp)    |
| 18001 | 18701 | `granite-plus`   | Plus model backend (PyTorch)      |

## Rationale

The original ports (8001, 8002, 9797) were not sequential and 8001/8002 sit
immediately adjacent to the heavily-used 8000 range, increasing the chance of
conflicts on shared dev or production machines.

The new public ports 8700–8702 are sequential, above the busy 8000–8099 band,
and below the OS ephemeral range (49152+), making conflicts unlikely.  Internal
ports follow the same base with a +10000 offset (18700, 18701), preserving the
visual relationship between a service and its backend.
