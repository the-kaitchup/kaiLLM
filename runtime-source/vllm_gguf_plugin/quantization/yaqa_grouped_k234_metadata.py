"""Pure metadata validation for the unselected original K2/K3/K4 core."""

MAX_PARTIAL_BYTES = 4 << 20


def descriptor_rows(shapes, bits, policies, rows, pointers):
    if not shapes or not len(shapes) == len(bits) == len(policies) == len(pointers):
        raise ValueError("Matching nonempty projection metadata required")
    if len(shapes) > 65535 or type(rows) is not int or not 1 <= rows <= 65535:
        raise ValueError("Bounded projection and physical row counts required")
    width = shapes[0][1]
    if type(width) is not int or not 0 < width <= 2**31 - 1 or width % 16:
        raise ValueError("Positive aligned input width required")
    ctas = offset = max_warps = 0
    descriptors = []
    for shape, bit, policy, ptrs in zip(shapes, bits, policies, pointers, strict=True):
        n, k = shape
        warps, prefetch, splits, shared = policy
        wptr, lptr = ptrs
        if (
            any(type(v) is not int for v in (n, k, bit, *policy, wptr, lptr))
            or k != width
            or n <= 0
            or n % 16
            or n > 2**31 - 1
            or bit not in (2, 3, 4)
            or n * k // 256 * (8 * bit) > 2**31 - 1
            or shared not in (0, 1)
            or warps not in ((4, 8, 16) if bit == 2 else (4, 8))
            or prefetch not in ((1, 2, 4) if bit == 2 else (2, 4, 8))
            or splits not in ((1,) if bit == 2 else (1, 2, 4, 8, 16, 32))
            or not 0 < wptr < 2**64
            or not 0 < lptr < 2**64
            or wptr % 4
            or lptr % 4
        ):
            raise ValueError("Invalid original bit-specific policy, wire or pointer")
        warp_rows = n // (2 if bit == 2 else 16)
        count = ((warp_rows + warps - 1) // warps) * splits
        descriptors.append(
            [wptr, lptr, n, splits, ctas, count, offset, bit, warps, prefetch, shared]
        )
        ctas += count
        offset += rows * splits * n
        max_warps = max(max_warps, warps)
    if ctas > 2**31 - 1 or offset * 4 > MAX_PARTIAL_BYTES:
        raise ValueError("Bounded grid or 4 MiB transient workspace exceeded")
    return descriptors, ctas, offset, max_warps
